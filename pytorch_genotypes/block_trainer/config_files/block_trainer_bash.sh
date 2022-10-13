#!/usr/bin/env bash

export N_CPUS={n_cpus}
export N_BLOCKS={n_blocks}

seq 0 $N_BLOCKS | xargs -n 1 -P $N_CPUS {command} train --chunk-index
