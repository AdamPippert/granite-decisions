"""Detached, fail-closed weekend launcher with an absolute process-group deadline."""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
import pwd
from pathlib import Path
import signal
import subprocess
import sys
import time

STOP=False


def request_stop(signum,frame):
    global STOP
    STOP=True


def write_json(path,value):
    path=Path(path);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def verify_inventory(root,filename):
    root=Path(root).resolve()
    for line in (root/filename).read_text().splitlines():
        expected,name=line.split('  ',1);path=(root/name).resolve()
        if not path.is_relative_to(root):raise ValueError('inventory path escapes run root')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('hash mismatch: '+name)


def check(root):
    root=Path(root)
    verify_inventory(root,'RUNTIME_SHA256SUMS')
    verify_inventory(root,'SHA256SUMS')
    config=json.loads((root/'schedule.json').read_text())
    if pwd.getpwuid(os.geteuid()).pw_name!=config['runtime_user']:
        raise ValueError('wrong runtime user')
    stamps=[datetime.fromisoformat(config[k]) for k in ('start','train_until','finish_until','hard_stop')]
    if any(x.tzinfo is None for x in stamps):raise ValueError('timezone required')
    if stamps!=sorted(set(stamps)):raise ValueError('invalid deadline order')
    pilot=json.loads((root/'pilot-single/pilot.json').read_text())
    if pilot.get('status')!='passed' or not pilot.get('final_evaluation_completed'):raise ValueError('GPU pilot not passed')
    if max(pilot['single_record_repeat_errors'])>1e-6:raise ValueError('GPU repeatability not passed')
    for name,version in json.loads((root/'environment.json').read_text()).items():
        if importlib.metadata.version(name)!=version:raise ValueError('runtime package changed: '+name)
    from weekend_data import load_data
    manifest,_=load_data(root/'data-v3')
    return config,manifest


def run_child(command,deadline,log_path,status_path,cancel_path,env=None,grace=5):
    """Kill the whole child session even if it ignores TERM; preserve exit evidence."""
    if time.time()>=deadline:raise ValueError('hard deadline already passed')
    with Path(log_path).open('ab',buffering=0) as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        write_json(status_path,{'phase':'running','pid':child.pid,'hard_stop_epoch':deadline})
        reason='exited'
        try:
            while child.poll() is None:
                if STOP or Path(cancel_path).exists():reason='cancelled';break
                if time.time()>=deadline-grace:reason='deadline';break
                time.sleep(min(.5,max(.01,deadline-grace-time.time())))
        finally:
            if child.poll() is None:
                try:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:child.wait(timeout=max(.01,min(grace,deadline-time.time())))
                except subprocess.TimeoutExpired:pass
            # A descendant may outlive a leader; clean the session in all cases.
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            child.wait(timeout=5)
        result={'phase':'finished' if reason=='exited' and child.returncode==0 else 'stopped',
            'reason':reason,'exit_code':child.returncode,'completed_utc':datetime.now(timezone.utc).isoformat()}
        write_json(status_path,result)
        return result


def validate_start(config,now):
    start=datetime.fromisoformat(config['start']).timestamp()
    if not start<=now<=start+config['maximum_start_delay_seconds']:
        raise ValueError('outside approved start window')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--check',action='store_true');args=p.parse_args()
    root=Path(args.root).resolve();os.chdir(root)
    lock=(root/'supervisor.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    config,manifest=check(root)
    if args.check:
        print(json.dumps({'status':'passed','config':config,'counts':manifest['counts']}));return
    validate_start(config,time.time())
    if (root/'run').exists():raise ValueError('run exists; refuse restart or overwrite')
    if (root/'CANCEL').exists():raise ValueError('run cancelled')
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    runtime=root/'runtime/experiments/weekend-20260918/scripts/train_weekend.py'
    command=[sys.executable,'-u',str(runtime),'--data',str(root/'data-v3'),'--out',str(root/'run'),
        '--steps',str(config['steps']),'--eval-every',str(config['eval_every']),
        '--train-until',config['train_until'],'--finish-until',config['finish_until']]
    env=os.environ.copy();env['HF_HUB_OFFLINE']='1';env['TOKENIZERS_PARALLELISM']='false'
    write_json(root/'launch.json',{'command':command,'config':config,'data_hashes':manifest['files'],
        'started_utc':datetime.now(timezone.utc).isoformat()})
    result=run_child(command,datetime.fromisoformat(config['hard_stop']).timestamp(),root/'training.log',root/'supervisor-status.json',root/'CANCEL',env)
    # Keep a compact local handoff even when a deadline/error prevents final evaluation.
    report={'supervisor':result,'automatic_weight_publication':False}
    for name in ('best','selection-frozen','report','status'):
        path=root/'run'/(name+'.json')
        if path.exists():report[name]=json.loads(path.read_text())
    write_json(root/'progress-summary.json',report)
    if result['phase']!='finished':raise SystemExit(1)


if __name__=='__main__':main()
