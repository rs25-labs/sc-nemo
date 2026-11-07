"""Offline API-adapter tests. Synthetic responses are not study judgments."""
import os
os.environ['DEEPEVAL_TELEMETRY_OPT_OUT']='1'
import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from evaluation.run_evaluation import score_job
ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=json.loads((ROOT/'evaluation/protocol.json').read_text())
class FakeClient:
 def __init__(self,fail=False):self.fail=fail;self.requests=[];self.chat=SimpleNamespace(completions=SimpleNamespace(create=self.create))
 def create(self,**request):
  self.requests.append(request)
  if self.fail:raise TimeoutError('Synthetic timeout')
  content=json.dumps({'score':8,'reason':'Synthetic offline transport response.'});d={'model':'synthetic-only','choices':[{'finish_reason':'stop','message':{'content':content}}]}
  return SimpleNamespace(model_dump=lambda **kwargs:d,choices=[SimpleNamespace(finish_reason='stop',message=SimpleNamespace(content=content))])
class JudgeTests(unittest.TestCase):
 def case(self):return {'blind_id':'synthetic-answer','input':'Synthetic expression evidence','actual_output':'Synthetic candidate','expected_output':'SYNTHETIC_EXPERT_REFERENCE'}
 def test_raw_score_normalization_and_expert_reference_boundary(self):
  with tempfile.TemporaryDirectory()as tmp:
   client=FakeClient()
   for spec in PROTOCOL['metrics']:
    r=score_job(self.case(),spec,client,PROTOCOL['judge'],Path(tmp),'synthetic-protocol','synthetic-key')
    self.assertEqual(r['status'],'complete');self.assertAlmostEqual(r['score'],.8)
    req=client.requests[-1];self.assertEqual(req['response_format']['type'],'json_schema')
    self.assertEqual('SYNTHETIC_EXPERT_REFERENCE'in req['messages'][1]['content'],spec['uses_expert_reference'])
 def test_failed_judge_has_no_invented_biological_score(self):
  with tempfile.TemporaryDirectory()as tmp:
   r=score_job(self.case(),PROTOCOL['metrics'][0],FakeClient(True),PROTOCOL['judge'],Path(tmp),'synthetic-protocol','synthetic-key')
   self.assertEqual(r['status'],'failed');self.assertIsNone(r['score'])
   self.assertEqual(len((Path(tmp)/'scores.jsonl').read_text().splitlines()),1)
if __name__=='__main__':unittest.main()
