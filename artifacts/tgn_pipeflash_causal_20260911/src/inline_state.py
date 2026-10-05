"""Enqueue outgoing state on the producer thread before independent work is submitted."""
def install():
    import modules.memory as mm
    def send_mem(self,mem,mail,rank,world_size,group=None,dst=-1):
        if mem is None:return None
        if dst==-1:dst=(rank+1)%world_size
        mm.send([mem,mail] if mem.shape[0]>0 else None,rank,dst,group)
        # GPU transfer remains asynchronous. There is no outstanding CPU launcher
        # thread for the next iteration to join.
        return None
    mm.Memory.send_mem=send_mem
