## Setting Up SSH

1. generate an `ssh-key` pair using `ssh-keygen` on you local machine
2. add the key *public* key to `~/.ssh/authorized_keys` on the `Dragan-DGX-Station`
3. configure host in your local ssh by adding this to your `~/.ssh/config` (on your own/local machine):
```
Host Dragan-DGX-Station
  HostName 128.32.41.243
  Port 1990
  User lassepe
  IdentityFile ~/.ssh/lassepe_interact
```

## How to use the scripts

- `path_definitions` is just a header that defines the relevant directories (from where to where to sync things)
- `connect` connects to the `Dragan-DGX-Station` via ssh and attaches to the `tmux` session
    - avoid closing the session. Use the detach session of `tmux`.
- `deploy_to_dgx` uses `rsync` to sync the local project folder to the `Dragan-DGX-Station`
- `downloadd_resutls` downloads the results from the remote machines `.../HumanSwitching.jl/resutls/` directory to the local machines `.../Humanswitching.jl/results/`
