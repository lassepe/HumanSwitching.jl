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
- `connect` connects to the `Dragan-DGX-Station` via ssh and attaches to the `tmux` session
    - avoid closing the session. Use the detach session of `tmux`.
- `deploy_to_dgx` uses `rsync` to sync the local project folder to the `Dragan-DGX-Station`
- `downloadd_resutls` downloads the results from the remote machines `.../HumanSwitching.jl/resutls/` directory to the local machines `.../HumanSwitching.jl/results/`

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

2. Create a file called `savio_username` in `.../HumanSwitching/bash_scripts/savio_config/` containing only your cluster user name (same as in step 1 for `<brc_user_name>`)

3. Configure `tmux` on your `brc` account
    - connect to your account via `ssh`. If you have completed `1.` this is as simple as typing `ssh SavioLogin`
    - if none existing: create a file called `.tmux.conf` and the configuration `new-session` in a separate line.
        This will make sure that upon login with `tmux`, a new session is created, if none exists yet.

4. Create the directory layout on the remote machine
    - connect to a Savio login node via `ssh`: `ssh SavioLogin`
    - on the remote machine, if non existing, create a directory `worktree` in your home directory. This will be used to place relevant files for simulation
