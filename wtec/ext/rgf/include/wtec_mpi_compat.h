#ifndef WTEC_MPI_COMPAT_H
#define WTEC_MPI_COMPAT_H

#ifdef WTEC_NO_MPI

typedef int MPI_Comm;
typedef int MPI_Datatype;
typedef int MPI_Op;

#define MPI_COMM_WORLD 0
#define MPI_SUCCESS 0
#define MPI_DOUBLE 1
#define MPI_INT 2
#define MPI_SUM 3
#define MPI_MAX 4

static inline int MPI_Init(int *argc, char ***argv) {
  (void)argc;
  (void)argv;
  return MPI_SUCCESS;
}
static inline int MPI_Finalize(void) { return MPI_SUCCESS; }
static inline int MPI_Initialized(int *flag) {
  if (flag != 0) {
    *flag = 1;
  }
  return MPI_SUCCESS;
}
static inline int MPI_Comm_rank(MPI_Comm comm, int *rank) {
  (void)comm;
  if (rank != 0) {
    *rank = 0;
  }
  return MPI_SUCCESS;
}
static inline int MPI_Comm_size(MPI_Comm comm, int *size) {
  (void)comm;
  if (size != 0) {
    *size = 1;
  }
  return MPI_SUCCESS;
}
static inline int MPI_Barrier(MPI_Comm comm) {
  (void)comm;
  return MPI_SUCCESS;
}
static inline int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm) {
  (void)buffer;
  (void)count;
  (void)datatype;
  (void)root;
  (void)comm;
  return MPI_SUCCESS;
}
static inline int MPI_Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm) {
  int i;
  (void)op;
  (void)root;
  (void)comm;
  if (sendbuf == recvbuf || recvbuf == 0 || sendbuf == 0) {
    return MPI_SUCCESS;
  }
  if (datatype == MPI_DOUBLE) {
    const double *src = (const double *)sendbuf;
    double *dst = (double *)recvbuf;
    for (i = 0; i < count; ++i) {
      dst[i] = src[i];
    }
  } else if (datatype == MPI_INT) {
    const int *src = (const int *)sendbuf;
    int *dst = (int *)recvbuf;
    for (i = 0; i < count; ++i) {
      dst[i] = src[i];
    }
  }
  return MPI_SUCCESS;
}

#else

#include <mpi.h>

#endif

#endif
