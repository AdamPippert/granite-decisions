"""Read live Codex quota without model inference; publish evidence only after reset.

Runs on superrouter. Unknown or unavailable quota leaves training paused.
No reset-credit consumption, API-key fallback or automatic spend.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import selectors
import subprocess
import time


def read_quota(timeout=20):
    child=subprocess.Popen(['codex','app-server','--stdio'],stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
    selector=selectors.DefaultSelector();selector.register(child.stdout,selectors.EVENT_READ)
    def send(value):child.stdin.write((json.dumps(value)+'\n').encode());child.stdin.flush()
    buffer=b''
    try:
        send({'id':1,'method':'initialize','params':{'clientInfo':{'name':'granite_reset_check','version':'1.0'}}})
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            if not selector.select(.5):continue
            block=os.read(child.stdout.fileno(),65536)
            if not block:break
            buffer+=block
            while b'\n' in buffer:
                line,buffer=buffer.split(b'\n',1);reply=json.loads(line)
                if reply.get('id') not in (1,2):continue
                if 'error' in reply:raise RuntimeError('quota service returned an error')
                if reply['id']==1:
                    send({'method':'initialized','params':{}})
                    send({'id':2,'method':'account/rateLimits/read'})
                else:
                    data=reply['result'];bucket=(data.get('rateLimitsByLimitId') or {}).get('codex') or data.get('rateLimits')
                    if not bucket or not data.get('accountId'):raise ValueError('missing account or quota bucket')
                    windows={k:bucket[k] for k in ('primary','secondary') if bucket.get(k)}
                    return {'account_hash':hashlib.sha256(data['accountId'].encode()).hexdigest(),
                        'ordinary_allowed':data.get('ordinaryUsageAllowed') is True,
                        'limit_reached':bucket.get('rateLimitReachedType'),
                        'spend_control_reached':bucket.get('spendControlReached',False),
                        'windows':windows,'checked_at':time.time()}
        raise TimeoutError('quota read timed out')
    finally:
        selector.close();child.terminate()
        try:child.wait(timeout=2)
        except subprocess.TimeoutExpired:child.kill();child.wait()


def reset_evidence(baseline,current,pause_at):
    now=current['checked_at']
    if now<pause_at or current['account_hash']!=baseline['account_hash']:return None
    if not current['ordinary_allowed'] or current['limit_reached'] or current['spend_control_reached']:return None
    windows=current['windows']
    if not windows or any(not 0<=w['usedPercent']<100 for w in windows.values()):return None
    for name,old in baseline['windows'].items():
        new=windows.get(name)
        if old['windowDurationMins']!=10080 or not new or new['windowDurationMins']!=10080:continue
        if old['resetsAt']<=now<new['resetsAt']:
            return {'reset_verified':True,'source':'account/rateLimits/read','checked_at':now,
                'previous_reset':old['resetsAt'],'current_reset':new['resetsAt'],
                'used_percent':new['usedPercent']}
    return None


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--pause-at',type=float,required=True)
    p.add_argument('--until',type=float,required=True);p.add_argument('--out',required=True);args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'resume-delivered.json').exists() or time.time()<args.pause_at or time.time()>=args.until:return
    try:
        current=read_quota();evidence=reset_evidence(json.loads(Path(args.baseline).read_text()),current,args.pause_at)
        (out/'quota-latest.json').write_text(json.dumps(current,indent=2)+'\n')
        if evidence is None:print('Reset not verified; training remains paused');return
        # The supervisor must actually have paused before this evidence releases it.
        ssh=['ssh','-F','/dev/null','-i','/home/adam/.ssh/homelab-hosts','-o','IdentitiesOnly=yes',
             '-o','StrictHostKeyChecking=yes','-o','BatchMode=yes','-o','ConnectTimeout=10','adam@hyde-fedora.tail4df14e.ts.net']
        root='/var/home/adam/Development/granite-decisions/experiments/weekend-20260918'
        result=subprocess.run(ssh+['cat '+root+'/supervisor-status.json'],capture_output=True,text=True,timeout=15,check=True)
        if json.loads(result.stdout).get('phase')!='paused':print('Waiting for training pause');return
        # Transfer through stdin and atomically rename; evidence contains no credentials.
        subprocess.run(ssh+['cat > '+root+'/resume-verified.json.tmp && mv '+root+'/resume-verified.json.tmp '+root+'/resume-verified.json'],
            input=json.dumps(evidence)+'\n',text=True,timeout=15,check=True)
        (out/'resume-delivered.json').write_text(json.dumps(evidence,indent=2)+'\n')
        print('Live reset verified; resume evidence delivered')
    except (OSError,ValueError,KeyError,TypeError,RuntimeError,TimeoutError,subprocess.SubprocessError) as error:
        print('Quota gate unavailable; training remains paused ('+type(error).__name__+')')
        raise SystemExit(1)


if __name__=='__main__':main()
