# DGX

## Setting Up SSH

1. generate an `ssh-key` pair using `ssh-keygen` on you local machine
2. add the key *public* key to `~/.ssh/authorized_keys` on the `Dragan-DGX-Station`
3. configure the host in your local ssh by adding this to your `~/.ssh/config` (on your own/local machine).  
   **Note:** Replace the placeholder `<ssh_key_idrsa>` with the file name of the private key whose public counterpart you have placed on the host.
```
Host Dragan-DGX-Station
  HostName 128.32.41.243
  Port 1990
  User lassepe
  IdentityFile ~/.ssh/<ssh_key_idrsa>
```

## How to use the scripts

- `header` is just a header that defines the relevant directories (from where to where to sync things) and other variables
- `connect_dgx` connects to the `Dragan-DGX-Station` via ssh and attaches to the `tmux` session
    - avoid closing the session. Use the detach session of `tmux`.
- `deploy_dgx` uses `rsync` to sync the local project folder to the `Dragan-DGX-Station`
- `download_dgx` downloads the results from the remote machines `./resutls/` directory to the local machines `./results/`

# Savio Cluster

## Setting Up a Savio User Account

- Follow the steps described here: <http://research-it.berkeley.edu/services/high-performance-computing/logging-savio>
- At the end of this process you should be able to manually `ssh` to a login node of the Savio cluster

## Configuration to Make Scripts Work With Savio

1. configure the host in your local ssh by adding this to your `~/.ssh/config` (on your own/local machine).  
   **Note:** Replace the placeholder `<brc_user_name>` with your cluster user name (NOT the group name `fc_hybrid`)
```
Host SavioLogin
  HostName hpc.brc.berkeley.edu
  User <brc_user_name>

Host SavioTransfer
  HostName dtn.brc.berkeley.edu
  User <brc_user_name>
```

2. Create a file called `savio_username` in `./bash_scripts/savio_config/` containing only your cluster user name (same as in step 1 for `<brc_user_name>`)

3. Configure `tmux` on your `brc` account
    - connect to your account via `ssh`. If you have completed `1.` this is as simple as typing `ssh SavioLogin`
    - if none existing: create a file called `.tmux.conf` and add the configuration `new-session` in a separate line.
        This will make sure that upon login with `tmux`, a new session is created, if none exists yet.

4. Create the directory layout on the remote machine
    - connect to a Savio login node via `ssh`: `ssh SavioLogin`
    - on the remote machine, if non existing, create a directory `worktree` in your home directory. This will be used to place relevant files for simulation

5. Setup `POMDPs.jl` registry on your Savio account
    - connect to Savio via ssh using `ssh SavioLogin`
    - load the Julia module: `module load julia/1.1.0`
    - setup `POMDPs.jl`
    ```julia
    using Pkg
    Pkg.add("POMDPs")
    using POMDPs
    POMDPs.add_registry()
    ```
6. Instantiate dependencies
    - **Note:** Technically this would happen automatically when you launch your first job. However, there is no need to waste compute units/credits by running this in batch mode. Therefore, we do it manually.
    - deploy the repository to Savio: On you local machine run the `./deploy_savio` script located in `./bash_scripts/`
    - connect to Savio via ssh using `ssh SavioLogin` and navigate to the project directory in `~/worktree/HumanSwitching.jl/` and
      start Julia in project mode `module load julia/1.1.0; julia --project`
    - from the Julia `REPL` run the following code (making sure that it finishes without errors)
    ```julia
    using Pkg
    Pkg.instantiate()
    ```

## Running Jobs on Savio

Now that everything has been setup, running experiments on Savio is very straight forward

1. Make sure the resources and time limit specified in `./bash_scripts/run_experiments.sbatch` match your needs.
2. Deploy to Savio using `./bash_scripts/deploy_savio`.
3. Connect to Savio via `ssh` using `ssh SavioLogin`.
4. Submit your job running `sbatch bash_scripts/run_experiments.sbatch` *form the project root*.

As soon as the job has finished you will receive an email if setup in
`./bash_scripts/run_experiments`. You can download the results using
`./bash_scripts/download_savio`. This will download any new `CSV` files and the
`slurm.out` log to your local `./results` directory.

## Useful Commands for Trouble Shooting and Monitoring

All of the following commands must be executed while being logged into Savio

- check usage of compute credits:
    - singe user: `check_usage.sh`
    - whole group: `check_usage -E -a fc_hybrid`
- view list of submitted jobs: `squeue -u $USER`
- cancelling a submitted job: `scancel <job-id>`
- check load on node to see whether your code uses all resources available:
    - option 1 (text based user interface, quit with `q`): `wwall -j $your_job_id -t`
    - option 2 (interactive terminal on node):
        - connect to node `srun --jobid=$your_job_id --pty /bin/bash`
        - run `htop`: `module load htop; htop`
