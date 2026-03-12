#ifndef WTEC_RGF_H
#define WTEC_RGF_H

#define WTEC_RGF_BINARY_ID "wtec_rgf_runner_phase2_v4"

typedef struct {
  int rank;
  int size;
  int mpi_enabled;
} wtec_rgf_probe_t;

int wtec_rgf_probe(wtec_rgf_probe_t *probe);

#endif
