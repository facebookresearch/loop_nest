# Benchmarks

Collections of benchmarks for loop_nest and other systems.

Each set of benchmarks is organized under a corresponding folder
of the form `<name>-benchmarks`. In that folder, there is a `run.sh`
bash script that can be run to execute all benchmarks. Each system
is executed using a separate `run_<system>.py` Python script.
Additional information for that set of benchmarks (including pending
TODOs) can be found in the corresponding `README.md`.

The `systems/` folder includes scripts (or instructions) to install
systems used to benchmark. Additionally, many of the systems used
expected particular environment variables, so we provide bash scripts
of the form `<system>_vars.sh` that can be sourced to set appropriate
variables. Please modify these to fit your machine setup, if you run
these on a separate machine (i.e. not the AWS machine used for NNC
benchmarking).

You can run all benchmarks by executing
`bash runall.sh`

We include some utility Python scripts to plot results
in the `utils/` folder. Please see `INSTALL-UTILS.md` for
instructions on setup.

