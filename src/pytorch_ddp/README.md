# PyTorch Distributed Data Parallel Example

The example is modified on [the example of PyTorch Tutorial](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html), referencing the following documents on DDP

* https://pytorch.org/tutorials/beginner/dist_overview.html
* https://pytorch.org/docs/stable/notes/ddp.html
* https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
* https://pytorch.org/tutorials/intermediate/dist_tuto.html

## How to run

* `run.py` is a basic example without parallel. It can be run with/without CUDA.
* `run_ddp.py` is a parallel example. It requires CUDA.

* For a command line help, run one with option `-h`.
* For `run_ddp.py`, an example command line could be

  ```bash
  python3 -m torch.distributed.run --nnodes=2 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=hostname:29400 /path/to/run_ddp.py -w 8
  ```

  Points to note here:
    * nnodes is the nubmer of nodes/VMs, on which `run_ddp.py` will run.
    * nproc_per_node is the number of processes per node/VM for `run_ddp.py`. One GPU can have only one such process. So if there are 4 GPUs on one node/VM, then the max nubmer for nproc_per_node is 4.
    * -w is the total number of paralell processes across all nodes. It must be nnodes * nproc_per_node.
    * hostname is the host name/IP of one node/VM among all nodes running `run_ddp.py`