"""Compare initial weights and report final-state differences without claiming trajectory identity."""
import json,torch,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',default='stable');args=parser.parse_args()
    data=json.loads((ROOT/'analysis'/(args.tag+'_summary.json')).read_text())
    by={(r['world_size'],r['variant'],r['seed']):r for r in data['records']}
    records=[]
    for world in [1,2,4]:
        for seed in [2026,2027,2028]:
            a=ROOT/'runs'/by[(world,'native',seed)]['label']/'results/rank0'
            b=ROOT/'runs'/by[(world,'flash',seed)]['label']/'results/rank0'
            load=lambda p:torch.load(p,map_location='cpu',weights_only=True)
            ia,ib=load(a/'initial_model.pt'),load(b/'initial_model.pt')
            fa,fb=load(a/'final_model.pt'),load(b/'final_model.pt')
            assert list(ia)==list(ib) and all(torch.equal(ia[k],ib[k]) for k in ia)
            num=sum(float((fa[k]-fb[k]).float().square().sum()) for k in fa)
            den=sum(float(fa[k].float().square().sum()) for k in fa)
            ma,mb=load(a/'final_memory.pt'),load(b/'final_memory.pt')
            records.append({'world_size':world,'seed':seed,'initial_weights_bitwise_equal':True,
                'final_weights_relative_l2':(num/den)**.5,
                'final_node_memory_timestamps_equal':torch.equal(ma['node_memory_ts'],mb['node_memory_ts']),
                'final_mailbox_timestamps_equal':torch.equal(ma['mailbox_ts'],mb['mailbox_ts'])})
    result={'records':records,'scope':'Initial rank-0 weight equality verified. Final-state differences are observations; no deterministic replay or sequential-optimizer equivalence is established.'}
    (ROOT/'analysis'/(args.tag+'_trajectory_observations.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if data['settings'].get('deterministic_last_event_mailbox'):
        assert all(x['final_node_memory_timestamps_equal'] and x['final_mailbox_timestamps_equal'] for x in records),'Timestamp mismatch despite last-event contract'
if __name__=='__main__':main()
