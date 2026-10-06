"""Standalone diagnostic endpoint timeline; connecting lines are not wire time."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    directory = args.root/'analysis/e2_endpoints'
    rows = [json.loads(line) for line in (directory/'matched_messages.jsonl').read_text().splitlines()]
    chosen = sorted([r for r in rows if 140 <= r['consumer_batch'] <= 147 and r['hash_verified']],
                    key=lambda r:(r['channel'], r['consumer_batch']))
    lo = min(min(r['payload_ready_gpu_ns'], r['receiver_host_call_begin_ns']) for r in chosen)
    fields = [('payload_ready_gpu_ns','D','#2070b4','Producer GPU ready'),
              ('sender_stream_ready_gpu_ns','^','#9259a3','Sender stream fence'),
              ('receiver_host_call_begin_ns','s','#dc854c','Receiver host call begins'),
              ('receive_ready_gpu_ns','o','#2b926e','Receiver stream ready')]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(13,8))
    for i,r in enumerate(chosen):
        begin = (r['payload_ready_gpu_ns']-lo)/1e6
        end = (r['receive_ready_gpu_ns']-lo)/1e6
        ax.plot([begin,end],[i,i],color='#d5d9dd',linewidth=2,zorder=1)
        for field,marker,color,label in fields:
            if r.get(field) is not None:
                ax.scatter((r[field]-lo)/1e6,i,marker=marker,color=color,s=30,
                           label=label if i==0 else None,zorder=3)
    ax.set_yticks(range(len(chosen)), [f"{r['channel']} b{r['payload_batch']} R{r['source_rank']} -> b{r['consumer_batch']} R{r['destination_rank']}" for r in chosen])
    ax.invert_yaxis()
    ax.set_xlabel('New diagnostic time (ms); common host clock with per-device CUDA anchors')
    ax.set_title('LastFM batches 140-147: matched payloads and observed endpoint fences\nLines include queueing and consumer scheduling; they are not pure communication latency')
    ax.grid(axis='x',alpha=.2)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.08),ncol=2,fontsize=9)
    fig.tight_layout()
    for extension in ['png','svg']:
        fig.savefig(directory/('endpoint_window_140_147.'+extension),dpi=150,bbox_inches='tight')
    plt.close(fig)
    (directory/'case_140_147.json').write_text(json.dumps({'messages':chosen,
        'parameter_versions':[json.loads(line) for line in (directory/'parameter_versions.jsonl').read_text().splitlines()
                               if 140 <= json.loads(line)['processing_batch'] <= 147],
        'pure_transport_latency_verified':False},indent=2)+'\n')
    print('Generated matched endpoint timeline and version case')


if __name__=='__main__': main()
