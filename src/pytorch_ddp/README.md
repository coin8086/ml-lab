# PyTorch Distributed Data Parallel Examples

The examples show how to use PyTorch DDP (Distributed Data Parallel):

* `run.py` is a basic example without parallel.
* `run_ddp.py` is a parallel example. It requires CUDA.
* `run_tut.py` is an example to show the principle of PyTorch DDP.

Reference:

* https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
* https://pytorch.org/tutorials/beginner/dist_overview.html
* https://pytorch.org/docs/stable/notes/ddp.html
* https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
* https://pytorch.org/tutorials/intermediate/dist_tuto.html
* https://pytorch.org/docs/stable/elastic/run.html

## How to run

* For a command line help, run one with option `-h`.
* For `run_ddp.py`, an example command line could be

  ```bash
  python3 -m torch.distributed.run --nnodes=2 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=hostname:29400 /path/to/run_ddp.py
  ```

  Points to note here:
    * `nnodes` is the number of nodes/VMs, on which `run_ddp.py` will run.
    * `nproc_per_node` is the number of processes per node/VM for `run_ddp.py`. In our example, one process has exclusive access to one GPU. So when there are 4 GPUs per node/VM, the max number for nproc_per_node is 4.
    * `rdzv_endpoint` hostname is the name/IP of one node/VM of all nodes running the command (process of rank 0 will be started on that host).