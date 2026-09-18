"""Generate original MIT-licensed synthetic fixtures, NOT a quality benchmark.
Templates are split as groups. Labels are assigned by construction, independently
of either model. Replace these with reviewed use-case data before making claims.
"""
import argparse
import itertools
import json
from pathlib import Path

QUESTIONS = {
 'route': {'type':'choice','instructions':'Which workflow does the request ask for?', 'criteria':{
     'code':'Implement or fix software.', 'retrieve':'Find existing information or documentation.', 'review':'Assess existing work without changing it.'}},
 'python': {'type':'noul','instructions':'Is Python the programming language explicitly named in the request?'},
 'scope': {'type':'score','instructions':'How broad is the explicitly stated scope of the work?',
           'criteria':['One function','One complete service','Several services']}}
TEMPLATES = {
 'train': {'code':['Implement the required changes in {scope} using {lang}.','Fix a bug in {scope} written in {lang}.','Write tests for {scope} in {lang}.'],
           'retrieve':['Find the documentation for {scope} written in {lang}.','Locate the source files for {scope} in {lang}.','Look up the API reference for {scope} implemented in {lang}.'] ,
           'review':['Review the design of {scope} in {lang}.','Assess the test coverage of {scope} written in {lang}.','Evaluate the readability of {scope} implemented in {lang}.']},
 'calibration': {'code':['Add input validation to {scope} written in {lang}.','Repair the error handling in {scope} using {lang}.','Implement logging for {scope} in {lang}.'],
           'retrieve':['Show me the existing documentation describing {scope} in {lang}.','Search for the deployment guide covering {scope} written in {lang}.','Retrieve the change history of {scope} implemented in {lang}.'],
           'review':['Audit {scope} written in {lang} for security problems without editing it.','Critique the proposed architecture of {scope} in {lang}.','Check whether {scope} in {lang} meets the stated requirements without changing it.']},
 'test': {'code':['Create a patch that corrects the failing tests in {scope} written in {lang}.','Develop the missing functionality for {scope} using {lang}.','Refactor {scope} in {lang} to remove duplicated logic.'],
           'retrieve':['Where is the published usage guide for {scope} in {lang}?','Identify the repository containing {scope} written in {lang}.','Fetch the release notes describing {scope} implemented in {lang}.'],
           'review':['Give feedback on the maintainability of {scope} in {lang}; leave the code unchanged.','Judge the quality of the proposed changes to {scope} written in {lang}.','Inspect {scope} in {lang} and report weaknesses without modifying anything.']}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True)
    args=p.parse_args()
    out=Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out/'questions.json').write_text(json.dumps(QUESTIONS,indent=2)+'\n')
    for split, groups in TEMPLATES.items():
        rows=[]
        for route, templates in groups.items():
            for template,(scope_id,scope),lang in itertools.product(templates,enumerate(['a single function','one complete service','several services']),['Python','JavaScript']):
                rows.append({'id':f'{split}-{len(rows):03d}', 'state':{'request':template.format(scope=scope,lang=lang)},
                             'labels':{'route':route,'python':lang=='Python','scope':scope_id}})
        (out/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (out/'README.txt').write_text('Synthetic integration fixtures only: 54 rows per split, 3 questions. Templates are grouped by split, but topics and vocabulary overlap. Not independent real-world evidence or a publication quality benchmark. Labels were assigned by construction. License: MIT.\n')

if __name__=='__main__':
    main()
