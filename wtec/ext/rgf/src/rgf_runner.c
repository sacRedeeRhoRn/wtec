#include "wtec_rgf.h"
#include "wtec_rgf_internal.h"
#include "wtec_mpi_compat.h"

#include <complex.h>
#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#else
static int omp_get_max_threads(void) { return 1; }
#endif

#ifndef WTEC_RGF_BUILD_BLAS_BACKEND
#define WTEC_RGF_BUILD_BLAS_BACKEND "none"
#endif

#define WTEC_RGF_MAX_PATH 4096
#define WTEC_RGF_JSON_EPS 1.0e-12
#define WTEC_RGF_SANCHO_MAX_ITER 256
#define WTEC_RGF_SANCHO_TOL 1.0e-10
#define WTEC_RGF_ETA_DEFAULT 1.0e-6
#define WTEC_RGF_TWO_PI 6.28318530717958647692

typedef struct {
  int num_wann;
  int n_r;
  int *rx;
  int *ry;
  int *rz;
  int *deg;
  double complex *mat;
} wtec_hr_model_t;

typedef struct {
  int n_terms;
  int *rx;
  int *rz;
  double complex *mat;
  int num_wann;
} wtec_ky_model_t;

typedef struct {
  char hr_dat_path[WTEC_RGF_MAX_PATH];
  char win_path[WTEC_RGF_MAX_PATH];
  char h_slices_path[WTEC_RGF_MAX_PATH];
  char v_slices_path[WTEC_RGF_MAX_PATH];
  char sigma_left_path[WTEC_RGF_MAX_PATH];
  char sigma_right_path[WTEC_RGF_MAX_PATH];
  char progress_file[WTEC_RGF_MAX_PATH];
  char queue[64];
  char logging_detail[32];
  char lead_axis;
  char thickness_axis;
  char periodic_axis;
  char mode[64];
  int *thicknesses;
  int n_thickness;
  int *mfp_lengths;
  int n_mfp_lengths;
  double *disorder_strengths;
  int n_disorder;
  int *slice_dims;
  int n_slice_dims;
  int n_layers_x;
  int n_layers_y;
  int mfp_n_layers_z;
  int n_ensemble;
  int base_seed;
  int expected_mpi_np;
  int heartbeat_seconds;
  double energy;
  double eta;
  int onsite_has_region;
  int onsite_x_start;
  int onsite_x_stop;
  int onsite_y_start;
  int onsite_y_stop;
  int onsite_z_start;
  int onsite_z_stop;
  double onsite_shift_ev;
} wtec_payload_t;

typedef struct {
  int p_eff;
  int slice_count;
  int superslice_dim;
  double transmission;
} wtec_point_result_t;

typedef struct {
  FILE *progress_fh;
  int rank;
  int size;
  int detail_per_step;
  int detail_per_ensemble;
  double heartbeat_seconds;
  double start_wall_s;
  double last_heartbeat_wall_s;
} wtec_progress_t;

typedef struct {
  int point_kind;
  int point_index;
  int point_total;
  int disorder_index;
  int ensemble_index;
  int sector_index;
  int sector_count;
  int nx;
  int ny;
  int nz;
  int seed;
  double disorder_strength;
} wtec_point_context_t;

typedef struct {
  int point_kind;
  int point_index;
  int point_total;
  int disorder_index;
  int ensemble_index;
  int sector_index;
  int sector_count;
  int nx;
  int ny;
  int nz;
  int seed;
  double disorder_strength;
} wtec_transport_task_t;

static double wtec_wall_seconds(void) {
  struct timeval tv;
  if (gettimeofday(&tv, NULL) != 0) {
    return (double)time(NULL);
  }
  return (double)tv.tv_sec + 1.0e-6 * (double)tv.tv_usec;
}

static const char *wtec_point_kind_name(int point_kind) {
  switch (point_kind) {
    case 0:
      return "thickness";
    case 1:
      return "length";
    default:
      return "block_validation";
  }
}

static const char *wtec_json_bool(int value) {
  return value ? "true" : "false";
}

static int wtec_effective_ensemble_count(double disorder_strength, int n_ensemble) {
  if (fabs(disorder_strength) <= WTEC_RGF_JSON_EPS) {
    return 1;
  }
  return (n_ensemble > 0) ? n_ensemble : 1;
}

static int wtec_mfp_disorder_index(const wtec_payload_t *payload) {
  if (payload == NULL || payload->n_disorder <= 0) {
    return 0;
  }
  return payload->n_disorder / 2;
}

static unsigned long long wtec_splitmix64(unsigned long long x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

static double wtec_disorder_sample(int seed, int gx, int gy, int gz, int orb) {
  unsigned long long x = (unsigned long long)(unsigned int)seed;
  x ^= 0xD1B54A32D192ED03ULL * (unsigned long long)(unsigned int)(gx + 0x9E37);
  x ^= 0x94D049BB133111EBULL * (unsigned long long)(unsigned int)(gy + 0x85EB);
  x ^= 0xBF58476D1CE4E5B9ULL * (unsigned long long)(unsigned int)(gz + 0xC2B2);
  x ^= 0xDB4F0B9175AE2165ULL * (unsigned long long)(unsigned int)(orb + 0x27D4);
  x = wtec_splitmix64(x);
  return ((double)(x >> 11) * (1.0 / 9007199254740992.0)) - 0.5;
}

static int wtec_progress_should_stdout(const wtec_progress_t *progress, const char *event) {
  if (progress == NULL || event == NULL) {
    return 0;
  }
  if (strcmp(event, "worker_start") == 0 ||
      strcmp(event, "transport_run_start") == 0 ||
      strcmp(event, "heartbeat") == 0 ||
      strcmp(event, "transport_run_done") == 0 ||
      strcmp(event, "worker_done") == 0 ||
      strcmp(event, "worker_failed") == 0) {
    return 1;
  }
  if (strcmp(event, "native_phase") == 0) {
    return progress->detail_per_step;
  }
  if (strcmp(event, "native_point_start") == 0 ||
      strcmp(event, "native_point_done") == 0) {
    return progress->detail_per_step || progress->detail_per_ensemble;
  }
  return 0;
}

static void wtec_progress_write_line(wtec_progress_t *progress, const char *line, int force_stdout) {
  if (line == NULL || line[0] == '\0') {
    return;
  }
  if (progress != NULL && progress->progress_fh != NULL && progress->rank == 0) {
    fputs(line, progress->progress_fh);
    fputc('\n', progress->progress_fh);
    fflush(progress->progress_fh);
  }
  if (force_stdout) {
    printf("[progress] %s\n", line);
    fflush(stdout);
  }
}

static void wtec_progress_emit_point_event(
    wtec_progress_t *progress,
    const char *event,
    const wtec_point_context_t *ctx,
    const char *phase,
    const wtec_point_result_t *result,
    double point_elapsed_s,
    int sweep_index,
    int sweep_total) {
  char line[2048];
  int off = 0;
  double now;
  if (progress == NULL || event == NULL) {
    return;
  }
  now = wtec_wall_seconds();
  off += snprintf(
      line + off,
      sizeof(line) - (size_t)off,
      "{\"ts\":%.6f,\"event\":\"%s\",\"rank\":%d,\"size\":%d,\"elapsed_s\":%.6f",
      now,
      event,
      progress->rank,
      progress->size,
      now - progress->start_wall_s);
  if (ctx != NULL) {
    off += snprintf(
        line + off,
        sizeof(line) - (size_t)off,
        ",\"point_kind\":\"%s\",\"point_index\":%d,\"point_total\":%d,"
        "\"disorder_index\":%d,\"disorder_strength\":%.16g,"
        "\"ensemble_index\":%d,\"seed\":%d,"
        "\"sector_index\":%d,\"sector_count\":%d,\"nx\":%d,\"ny\":%d,\"nz\":%d",
        wtec_point_kind_name(ctx->point_kind),
        ctx->point_index,
        ctx->point_total,
        ctx->disorder_index,
        ctx->disorder_strength,
        ctx->ensemble_index,
        ctx->seed,
        ctx->sector_index,
        ctx->sector_count,
        ctx->nx,
        ctx->ny,
        ctx->nz);
  }
  if (phase != NULL && phase[0] != '\0') {
    off += snprintf(
        line + off,
        sizeof(line) - (size_t)off,
        ",\"phase\":\"%s\"",
        phase);
  }
  if (sweep_index >= 0 && sweep_total > 0) {
    off += snprintf(
        line + off,
        sizeof(line) - (size_t)off,
        ",\"sweep_index\":%d,\"sweep_total\":%d",
        sweep_index,
        sweep_total);
  }
  if (result != NULL) {
    off += snprintf(
        line + off,
        sizeof(line) - (size_t)off,
        ",\"p_eff\":%d,\"slice_count\":%d,\"superslice_dim\":%d,\"G\":%.16g",
        result->p_eff,
        result->slice_count,
        result->superslice_dim,
        result->transmission);
  }
  if (point_elapsed_s >= 0.0) {
    off += snprintf(
        line + off,
        sizeof(line) - (size_t)off,
        ",\"point_elapsed_s\":%.6f",
        point_elapsed_s);
  }
  snprintf(line + off, sizeof(line) - (size_t)off, "}");
  wtec_progress_write_line(progress, line, wtec_progress_should_stdout(progress, event));
}

static void wtec_progress_emit_runner_event(
    wtec_progress_t *progress,
    const char *event,
    const wtec_payload_t *payload,
    int transport_task_count) {
  char line[1024];
  int off = 0;
  double now;
  if (progress == NULL || event == NULL || payload == NULL) {
    return;
  }
  now = wtec_wall_seconds();
  off += snprintf(
      line + off,
      sizeof(line) - (size_t)off,
      "{\"ts\":%.6f,\"event\":\"%s\",\"rank\":%d,\"size\":%d,\"elapsed_s\":%.6f,"
      "\"mode\":\"%s\",\"n_thickness\":%d,\"n_mfp_lengths\":%d,\"n_disorder\":%d,"
      "\"n_ensemble\":%d,\"transport_task_count\":%d,"
      "\"logging_detail\":\"%s\",\"heartbeat_seconds\":%.6f}",
      now,
      event,
      progress->rank,
      progress->size,
      now - progress->start_wall_s,
      payload->mode,
      payload->n_thickness,
      payload->n_mfp_lengths,
      payload->n_disorder,
      payload->n_ensemble,
      transport_task_count,
      payload->logging_detail,
      progress->heartbeat_seconds);
  wtec_progress_write_line(progress, line, wtec_progress_should_stdout(progress, event));
}

static void wtec_progress_maybe_heartbeat(
    wtec_progress_t *progress,
    const wtec_point_context_t *ctx,
    const char *phase) {
  double now;
  if (progress == NULL || progress->heartbeat_seconds <= 0.0) {
    return;
  }
  now = wtec_wall_seconds();
  if ((now - progress->last_heartbeat_wall_s) < progress->heartbeat_seconds) {
    return;
  }
  progress->last_heartbeat_wall_s = now;
  wtec_progress_emit_point_event(progress, "heartbeat", ctx, phase, NULL, -1.0, -1, -1);
}

static void wtec_progress_step(
    wtec_progress_t *progress,
    const wtec_point_context_t *ctx,
    const char *phase,
    int sweep_index,
    int sweep_total) {
  if (progress == NULL) {
    return;
  }
  if (progress->detail_per_step) {
    wtec_progress_emit_point_event(progress, "native_phase", ctx, phase, NULL, -1.0, sweep_index, sweep_total);
  }
  wtec_progress_maybe_heartbeat(progress, ctx, phase);
}

static void wtec_progress_init(
    wtec_progress_t *progress,
    const wtec_payload_t *payload,
    int rank,
    int size) {
  if (progress == NULL || payload == NULL) {
    return;
  }
  memset(progress, 0, sizeof(*progress));
  progress->rank = rank;
  progress->size = size;
  progress->detail_per_step = strcmp(payload->logging_detail, "per_step") == 0;
  progress->detail_per_ensemble =
      strcmp(payload->logging_detail, "per_ensemble") == 0 || progress->detail_per_step;
  progress->heartbeat_seconds = (payload->heartbeat_seconds > 0)
      ? (double)payload->heartbeat_seconds
      : 20.0;
  progress->start_wall_s = wtec_wall_seconds();
  progress->last_heartbeat_wall_s = progress->start_wall_s;
  if (rank == 0 && payload->progress_file[0] != '\0') {
    progress->progress_fh = fopen(payload->progress_file, "w");
    if (progress->progress_fh == NULL) {
      fprintf(stderr, "failed to open progress file %s: %s\n", payload->progress_file, strerror(errno));
    }
  }
}

static void wtec_progress_close(wtec_progress_t *progress) {
  if (progress == NULL) {
    return;
  }
  if (progress->progress_fh != NULL) {
    fclose(progress->progress_fh);
    progress->progress_fh = NULL;
  }
}

static void *wtec_calloc(size_t n, size_t size) {
  void *ptr = calloc(n, size);
  if (ptr == NULL) {
    fprintf(stderr, "allocation failed for %zu x %zu bytes\n", n, size);
    exit(2);
  }
  return ptr;
}

static char *wtec_read_text_file(const char *path) {
  FILE *fh = fopen(path, "rb");
  long size;
  char *buf;
  if (fh == NULL) {
    fprintf(stderr, "failed to open %s: %s\n", path, strerror(errno));
    return NULL;
  }
  if (fseek(fh, 0L, SEEK_END) != 0) {
    fclose(fh);
    return NULL;
  }
  size = ftell(fh);
  if (size < 0) {
    fclose(fh);
    return NULL;
  }
  if (fseek(fh, 0L, SEEK_SET) != 0) {
    fclose(fh);
    return NULL;
  }
  buf = (char *)wtec_calloc((size_t)size + 1u, sizeof(char));
  if ((long)fread(buf, 1u, (size_t)size, fh) != size) {
    fclose(fh);
    free(buf);
    return NULL;
  }
  fclose(fh);
  buf[size] = '\0';
  return buf;
}

static const char *wtec_skip_ws(const char *p) {
  while (p != NULL && *p != '\0' && isspace((unsigned char)*p)) {
    ++p;
  }
  return p;
}

static const char *wtec_find_json_key(const char *text, const char *key) {
  char pattern[256];
  char *found;
  snprintf(pattern, sizeof(pattern), "\"%s\"", key);
  found = strstr(text, pattern);
  if (found == NULL) {
    return NULL;
  }
  found = strchr(found + (long)strlen(pattern), ':');
  if (found == NULL) {
    return NULL;
  }
  return wtec_skip_ws(found + 1);
}

static int wtec_parse_json_string(const char *text, const char *key, char *out, size_t out_sz) {
  const char *p = wtec_find_json_key(text, key);
  size_t i = 0;
  if (p == NULL || *p == 'n') {
    if (out_sz > 0u) {
      out[0] = '\0';
    }
    return 0;
  }
  if (*p != '"') {
    return -1;
  }
  ++p;
  while (*p != '\0' && *p != '"' && i + 1u < out_sz) {
    if (*p == '\\' && p[1] != '\0') {
      ++p;
    }
    out[i++] = *p++;
  }
  if (*p != '"') {
    return -1;
  }
  out[i] = '\0';
  return 0;
}

static int wtec_parse_json_double(const char *text, const char *key, double *out) {
  const char *p = wtec_find_json_key(text, key);
  char *endptr;
  if (p == NULL) {
    return -1;
  }
  *out = strtod(p, &endptr);
  return (endptr == p) ? -1 : 0;
}

static int wtec_parse_json_int(const char *text, const char *key, int *out) {
  double tmp = 0.0;
  if (wtec_parse_json_double(text, key, &tmp) != 0) {
    return -1;
  }
  *out = (int)llround(tmp);
  return 0;
}

static int wtec_parse_json_int_array(const char *text, const char *key, int **out, int *n_out) {
  const char *p = wtec_find_json_key(text, key);
  int count = 0;
  int cap = 8;
  int *vals;
  if (p == NULL || *p == 'n') {
    *out = NULL;
    *n_out = 0;
    return 0;
  }
  if (*p != '[') {
    return -1;
  }
  vals = (int *)wtec_calloc((size_t)cap, sizeof(int));
  ++p;
  for (;;) {
    char *endptr;
    double value;
    p = wtec_skip_ws(p);
    if (*p == ']') {
      ++p;
      break;
    }
    value = strtod(p, &endptr);
    if (endptr == p) {
      free(vals);
      return -1;
    }
    if (count >= cap) {
      cap *= 2;
      vals = (int *)realloc(vals, (size_t)cap * sizeof(int));
      if (vals == NULL) {
        fprintf(stderr, "realloc failed\n");
        exit(2);
      }
    }
    vals[count++] = (int)llround(value);
    p = wtec_skip_ws(endptr);
    if (*p == ',') {
      ++p;
      continue;
    }
    if (*p == ']') {
      ++p;
      break;
    }
    free(vals);
    return -1;
  }
  *out = vals;
  *n_out = count;
  return 0;
}

static int wtec_parse_json_double_array(const char *text, const char *key, double **out, int *n_out) {
  const char *p = wtec_find_json_key(text, key);
  int count = 0;
  int cap = 8;
  double *vals;
  if (p == NULL || *p == 'n') {
    *out = NULL;
    *n_out = 0;
    return 0;
  }
  if (*p != '[') {
    return -1;
  }
  vals = (double *)wtec_calloc((size_t)cap, sizeof(double));
  ++p;
  for (;;) {
    char *endptr;
    double value;
    p = wtec_skip_ws(p);
    if (*p == ']') {
      ++p;
      break;
    }
    value = strtod(p, &endptr);
    if (endptr == p) {
      free(vals);
      return -1;
    }
    if (count >= cap) {
      cap *= 2;
      vals = (double *)realloc(vals, (size_t)cap * sizeof(double));
      if (vals == NULL) {
        fprintf(stderr, "realloc failed\n");
        exit(2);
      }
    }
    vals[count++] = value;
    p = wtec_skip_ws(endptr);
    if (*p == ',') {
      ++p;
      continue;
    }
    if (*p == ']') {
      ++p;
      break;
    }
    free(vals);
    return -1;
  }
  *out = vals;
  *n_out = count;
  return 0;
}

static void wtec_free_payload(wtec_payload_t *payload) {
  if (payload == NULL) {
    return;
  }
  free(payload->thicknesses);
  free(payload->mfp_lengths);
  free(payload->disorder_strengths);
  free(payload->slice_dims);
  memset(payload, 0, sizeof(*payload));
}

static int wtec_parse_payload_file(const char *path, wtec_payload_t *payload, char *err, size_t err_sz) {
  char *text = wtec_read_text_file(path);
  int i;
  if (payload == NULL) {
    return -1;
  }
  memset(payload, 0, sizeof(*payload));
  if (text == NULL) {
    snprintf(err, err_sz, "failed to read payload file: %s", path);
    return -1;
  }
  if (wtec_parse_json_string(text, "hr_dat_path", payload->hr_dat_path, sizeof(payload->hr_dat_path)) != 0 ||
      payload->hr_dat_path[0] == '\0') {
    snprintf(err, err_sz, "payload missing hr_dat_path");
    free(text);
    return -1;
  }
  (void)wtec_parse_json_string(text, "win_path", payload->win_path, sizeof(payload->win_path));
  (void)wtec_parse_json_string(text, "progress_file", payload->progress_file, sizeof(payload->progress_file));
  (void)wtec_parse_json_string(text, "queue", payload->queue, sizeof(payload->queue));
  if (wtec_parse_json_string(text, "logging_detail", payload->logging_detail, sizeof(payload->logging_detail)) != 0 ||
      payload->logging_detail[0] == '\0') {
    snprintf(payload->logging_detail, sizeof(payload->logging_detail), "minimal");
  }
  if (wtec_parse_json_int(text, "heartbeat_seconds", &payload->heartbeat_seconds) != 0 ||
      payload->heartbeat_seconds <= 0) {
    payload->heartbeat_seconds = 20;
  }
  if (wtec_parse_json_int(text, "n_layers_x", &payload->n_layers_x) != 0 ||
      wtec_parse_json_int(text, "n_layers_y", &payload->n_layers_y) != 0 ||
      wtec_parse_json_int(text, "mfp_n_layers_z", &payload->mfp_n_layers_z) != 0) {
    snprintf(err, err_sz, "payload missing required geometry integers");
    free(text);
    return -1;
  }
  if (wtec_parse_json_int(text, "n_ensemble", &payload->n_ensemble) != 0 ||
      payload->n_ensemble <= 0) {
    payload->n_ensemble = 1;
  }
  (void)wtec_parse_json_int(text, "base_seed", &payload->base_seed);
  if (wtec_parse_json_double(text, "energy", &payload->energy) != 0) {
    payload->energy = 0.0;
  }
  if (wtec_parse_json_double(text, "eta", &payload->eta) != 0) {
    payload->eta = WTEC_RGF_ETA_DEFAULT;
  }
  (void)wtec_parse_json_double(text, "onsite_region_shift_ev", &payload->onsite_shift_ev);
  (void)wtec_parse_json_int(text, "onsite_region_x_start_uc", &payload->onsite_x_start);
  (void)wtec_parse_json_int(text, "onsite_region_x_stop_uc", &payload->onsite_x_stop);
  (void)wtec_parse_json_int(text, "onsite_region_y_start_uc", &payload->onsite_y_start);
  (void)wtec_parse_json_int(text, "onsite_region_y_stop_uc", &payload->onsite_y_stop);
  (void)wtec_parse_json_int(text, "onsite_region_z_start_uc", &payload->onsite_z_start);
  (void)wtec_parse_json_int(text, "onsite_region_z_stop_uc", &payload->onsite_z_stop);
  if (fabs(payload->onsite_shift_ev) > WTEC_RGF_JSON_EPS) {
    payload->onsite_has_region = 1;
  }
  (void)wtec_parse_json_int(text, "expected_mpi_np", &payload->expected_mpi_np);
  if (wtec_parse_json_int_array(text, "thicknesses", &payload->thicknesses, &payload->n_thickness) != 0 ||
      wtec_parse_json_int_array(text, "mfp_lengths", &payload->mfp_lengths, &payload->n_mfp_lengths) != 0 ||
      wtec_parse_json_double_array(text, "disorder_strengths", &payload->disorder_strengths, &payload->n_disorder) != 0 ||
      wtec_parse_json_int_array(text, "slice_dims", &payload->slice_dims, &payload->n_slice_dims) != 0) {
    snprintf(err, err_sz, "payload array parse failed");
    free(text);
    wtec_free_payload(payload);
    return -1;
  }
  (void)wtec_parse_json_string(text, "h_slices_path", payload->h_slices_path, sizeof(payload->h_slices_path));
  (void)wtec_parse_json_string(text, "v_slices_path", payload->v_slices_path, sizeof(payload->v_slices_path));
  (void)wtec_parse_json_string(text, "sigma_left_path", payload->sigma_left_path, sizeof(payload->sigma_left_path));
  (void)wtec_parse_json_string(text, "sigma_right_path", payload->sigma_right_path, sizeof(payload->sigma_right_path));
  if (wtec_parse_json_string(text, "lead_axis", payload->mode, sizeof(payload->mode)) != 0 ||
      payload->mode[0] == '\0') {
    snprintf(err, err_sz, "payload missing lead_axis");
    free(text);
    wtec_free_payload(payload);
    return -1;
  }
  payload->lead_axis = payload->mode[0];
  if (wtec_parse_json_string(text, "thickness_axis", payload->mode, sizeof(payload->mode)) != 0 ||
      payload->mode[0] == '\0') {
    snprintf(err, err_sz, "payload missing thickness_axis");
    free(text);
    wtec_free_payload(payload);
    return -1;
  }
  payload->thickness_axis = payload->mode[0];
  if (wtec_parse_json_string(text, "transport_engine", payload->mode, sizeof(payload->mode)) != 0 ||
      strcmp(payload->mode, "rgf") != 0) {
    snprintf(err, err_sz, "payload transport_engine must be 'rgf'");
    free(text);
    wtec_free_payload(payload);
    return -1;
  }
  if (wtec_parse_json_string(text, "transport_rgf_mode", payload->mode, sizeof(payload->mode)) != 0 &&
      wtec_parse_json_string(text, "rgf_mode", payload->mode, sizeof(payload->mode)) != 0) {
    snprintf(err, err_sz, "payload missing rgf_mode");
    free(text);
    wtec_free_payload(payload);
    return -1;
  }
  if (wtec_parse_json_string(text, "transport_rgf_periodic_axis", payload->queue, sizeof(payload->queue)) != 0 &&
      wtec_parse_json_string(text, "rgf_periodic_axis", payload->queue, sizeof(payload->queue)) != 0) {
    payload->queue[0] = '\0';
  }
  payload->periodic_axis = payload->queue[0];
  payload->queue[0] = '\0';
  (void)wtec_parse_json_string(text, "queue", payload->queue, sizeof(payload->queue));
  free(text);

  if (strcmp(payload->mode, "periodic_transverse") != 0 &&
      strcmp(payload->mode, "full_finite") != 0 &&
      strcmp(payload->mode, "block_validation") != 0) {
    snprintf(err, err_sz, "unsupported rgf_mode=%s", payload->mode);
    wtec_free_payload(payload);
    return -1;
  }
  if (payload->lead_axis != 'x') {
    snprintf(err, err_sz, "current native RGF supports lead_axis='x' only");
    wtec_free_payload(payload);
    return -1;
  }
  if (strcmp(payload->mode, "periodic_transverse") == 0 &&
      payload->periodic_axis != 'y') {
    snprintf(err, err_sz, "phase1 RGF supports lead_axis='x' and periodic_axis='y' only");
    wtec_free_payload(payload);
    return -1;
  }
  if (payload->thickness_axis != 'z') {
    snprintf(err, err_sz, "current native RGF supports thickness_axis='z' only");
    wtec_free_payload(payload);
    return -1;
  }
  if (strcmp(payload->mode, "block_validation") == 0) {
    if (payload->h_slices_path[0] == '\0' || payload->v_slices_path[0] == '\0' ||
        payload->sigma_left_path[0] == '\0' || payload->sigma_right_path[0] == '\0' ||
        payload->n_slice_dims <= 0) {
      snprintf(err, err_sz, "block_validation requires h_slices_path, v_slices_path, sigma_left_path, sigma_right_path and slice_dims");
      wtec_free_payload(payload);
      return -1;
    }
  }
  if (payload->n_layers_x <= 0 || payload->n_layers_y <= 0 || payload->mfp_n_layers_z <= 0) {
    snprintf(err, err_sz, "geometry values must be > 0");
    wtec_free_payload(payload);
    return -1;
  }
  if (payload->n_disorder <= 0 || payload->disorder_strengths == NULL) {
    snprintf(err, err_sz, "disorder_strengths must contain at least one value");
    wtec_free_payload(payload);
    return -1;
  }
  if (payload->n_ensemble <= 0) {
    snprintf(err, err_sz, "n_ensemble must be > 0");
    wtec_free_payload(payload);
    return -1;
  }
  for (i = 0; i < payload->n_disorder; ++i) {
    if (strcmp(payload->mode, "periodic_transverse") == 0 &&
        fabs(payload->disorder_strengths[i]) > WTEC_RGF_JSON_EPS) {
      snprintf(err, err_sz, "phase1 RGF supports clean transport only");
      wtec_free_payload(payload);
      return -1;
    }
  }
  if (payload->onsite_has_region) {
    if (payload->onsite_x_stop <= payload->onsite_x_start ||
        payload->onsite_y_stop <= payload->onsite_y_start ||
        payload->onsite_z_stop <= payload->onsite_z_start) {
      snprintf(err, err_sz, "onsite region bounds must satisfy stop > start for all axes");
      wtec_free_payload(payload);
      return -1;
    }
  }
  return 0;
}

static void wtec_free_hr_model(wtec_hr_model_t *model) {
  if (model == NULL) {
    return;
  }
  free(model->rx);
  free(model->ry);
  free(model->rz);
  free(model->deg);
  free(model->mat);
  memset(model, 0, sizeof(*model));
}

static int wtec_parse_hr_file(const char *path, wtec_hr_model_t *model, char *err, size_t err_sz) {
  FILE *fh;
  char line[4096];
  int n_r, num_wann, ri, row_count, deg_index;
  if (model == NULL) {
    return -1;
  }
  memset(model, 0, sizeof(*model));
  fh = fopen(path, "r");
  if (fh == NULL) {
    snprintf(err, err_sz, "failed to open hr file %s: %s", path, strerror(errno));
    return -1;
  }
  if (fgets(line, sizeof(line), fh) == NULL ||
      fgets(line, sizeof(line), fh) == NULL) {
    fclose(fh);
    snprintf(err, err_sz, "hr file truncated");
    return -1;
  }
  num_wann = atoi(line);
  if (fgets(line, sizeof(line), fh) == NULL) {
    fclose(fh);
    snprintf(err, err_sz, "hr file missing n_R");
    return -1;
  }
  n_r = atoi(line);
  if (num_wann <= 0 || n_r <= 0) {
    fclose(fh);
    snprintf(err, err_sz, "invalid hr header");
    return -1;
  }
  model->num_wann = num_wann;
  model->n_r = n_r;
  model->rx = (int *)wtec_calloc((size_t)n_r, sizeof(int));
  model->ry = (int *)wtec_calloc((size_t)n_r, sizeof(int));
  model->rz = (int *)wtec_calloc((size_t)n_r, sizeof(int));
  model->deg = (int *)wtec_calloc((size_t)n_r, sizeof(int));
  model->mat = (double complex *)wtec_calloc((size_t)n_r * (size_t)num_wann * (size_t)num_wann, sizeof(double complex));

  row_count = 0;
  deg_index = 0;
  while (row_count < n_r) {
    char *p;
    if (fgets(line, sizeof(line), fh) == NULL) {
      fclose(fh);
      snprintf(err, err_sz, "hr file truncated in degeneracy block");
      wtec_free_hr_model(model);
      return -1;
    }
    p = line;
    while (*p != '\0') {
      char *endptr;
      long deg_val = strtol(p, &endptr, 10);
      if (endptr == p) {
        break;
      }
      if (deg_index < n_r) {
        model->deg[deg_index++] = (int)(deg_val > 0 ? deg_val : 1);
      }
      ++row_count;
      p = endptr;
      if (row_count >= n_r) {
        break;
      }
    }
  }

  ri = -1;
  while (fgets(line, sizeof(line), fh) != NULL) {
    int rx, ry, rz, n, m;
    double re_val, im_val;
    char *trim = line;
    while (*trim != '\0' && isspace((unsigned char)*trim)) {
      ++trim;
    }
    if (*trim == '\0') {
      continue;
    }
    if (sscanf(trim, "%d %d %d %d %d %lf %lf", &rx, &ry, &rz, &n, &m, &re_val, &im_val) != 7) {
      continue;
    }
    if (ri < 0 || model->rx[ri] != rx || model->ry[ri] != ry || model->rz[ri] != rz) {
      ++ri;
      if (ri >= n_r) {
        break;
      }
      model->rx[ri] = rx;
      model->ry[ri] = ry;
      model->rz[ri] = rz;
    }
    if (n <= 0 || n > num_wann || m <= 0 || m > num_wann) {
      fclose(fh);
      snprintf(err, err_sz, "hr matrix index out of range");
      wtec_free_hr_model(model);
      return -1;
    }
    model->mat[((size_t)ri * (size_t)num_wann + (size_t)(m - 1)) * (size_t)num_wann + (size_t)(n - 1)] =
        re_val + I * im_val;
  }
  fclose(fh);
  if (ri + 1 < n_r) {
    model->n_r = ri + 1;
  }
  for (ri = 0; ri < model->n_r; ++ri) {
    if (model->deg[ri] <= 0) {
      model->deg[ri] = 1;
    }
  }
  return 0;
}

static void wtec_free_ky_model(wtec_ky_model_t *ky) {
  if (ky == NULL) {
    return;
  }
  free(ky->rx);
  free(ky->rz);
  free(ky->mat);
  memset(ky, 0, sizeof(*ky));
}

static int wtec_find_ky_term(const wtec_ky_model_t *ky, int rx, int rz) {
  int i;
  for (i = 0; i < ky->n_terms; ++i) {
    if (ky->rx[i] == rx && ky->rz[i] == rz) {
      return i;
    }
  }
  return -1;
}

static int wtec_build_ky_model(const wtec_hr_model_t *model, double ky_frac, wtec_ky_model_t *ky) {
  int i, idx;
  int cap = model->n_r > 0 ? model->n_r : 1;
  const int nw = model->num_wann;
  const double two_pi = WTEC_RGF_TWO_PI;
  memset(ky, 0, sizeof(*ky));
  ky->num_wann = nw;
  ky->rx = (int *)wtec_calloc((size_t)cap, sizeof(int));
  ky->rz = (int *)wtec_calloc((size_t)cap, sizeof(int));
  ky->mat = (double complex *)wtec_calloc((size_t)cap * (size_t)nw * (size_t)nw, sizeof(double complex));
  for (i = 0; i < model->n_r; ++i) {
    double angle = two_pi * ky_frac * (double)model->ry[i];
    double complex phase = cos(angle) + I * sin(angle);
    double inv_deg = 1.0 / (double)((model->deg != NULL && model->deg[i] > 0) ? model->deg[i] : 1);
    idx = wtec_find_ky_term(ky, model->rx[i], model->rz[i]);
    if (idx < 0) {
      idx = ky->n_terms++;
      ky->rx[idx] = model->rx[i];
      ky->rz[idx] = model->rz[i];
    }
    {
      size_t j;
      double complex *dst = ky->mat + (size_t)idx * (size_t)nw * (size_t)nw;
      const double complex *src = model->mat + (size_t)i * (size_t)nw * (size_t)nw;
      for (j = 0; j < (size_t)nw * (size_t)nw; ++j) {
        dst[j] += inv_deg * phase * src[j];
      }
    }
  }
  return 0;
}

static int wtec_p_eff_for_nz(const wtec_ky_model_t *ky, int nz) {
  int i;
  int max_abs = 1;
  for (i = 0; i < ky->n_terms; ++i) {
    if (abs(ky->rz[i]) > nz - 1) {
      continue;
    }
    if (abs(ky->rx[i]) > max_abs) {
      max_abs = abs(ky->rx[i]);
    }
  }
  return max_abs < 1 ? 1 : max_abs;
}

static int wtec_p_eff_for_model_nz(const wtec_hr_model_t *model, int nz) {
  int i;
  int max_abs = 1;
  for (i = 0; i < model->n_r; ++i) {
    if (abs(model->rz[i]) > nz - 1) {
      continue;
    }
    if (abs(model->rx[i]) > max_abs) {
      max_abs = abs(model->rx[i]);
    }
  }
  return max_abs < 1 ? 1 : max_abs;
}

static int wtec_p_eff_for_model_full(const wtec_hr_model_t *model, int ny, int nz) {
  int i;
  int max_abs = 1;
  for (i = 0; i < model->n_r; ++i) {
    if (abs(model->ry[i]) > ny - 1 || abs(model->rz[i]) > nz - 1) {
      continue;
    }
    if (abs(model->rx[i]) > max_abs) {
      max_abs = abs(model->rx[i]);
    }
  }
  return max_abs < 1 ? 1 : max_abs;
}

static void wtec_active_transverse_spans_full(
    const wtec_hr_model_t *model,
    int ny,
    int nz,
    int *max_ry_out,
    int *max_rz_out) {
  int i;
  int max_ry = 0;
  int max_rz = 0;
  for (i = 0; i < model->n_r; ++i) {
    if (abs(model->ry[i]) > ny - 1 || abs(model->rz[i]) > nz - 1) {
      continue;
    }
    if (abs(model->ry[i]) > max_ry) {
      max_ry = abs(model->ry[i]);
    }
    if (abs(model->rz[i]) > max_rz) {
      max_rz = abs(model->rz[i]);
    }
  }
  if (max_ry_out != NULL) *max_ry_out = max_ry;
  if (max_rz_out != NULL) *max_rz_out = max_rz;
}

static double complex *wtec_mat_alloc(int rows, int cols) {
  return (double complex *)wtec_calloc((size_t)rows * (size_t)cols, sizeof(double complex));
}

static void wtec_mat_conj_transpose(const double complex *a, int rows, int cols, double complex *out) {
  int i, j;
#if defined(_OPENMP)
#pragma omp parallel for private(j) schedule(static) if ((rows * cols) >= 4096)
#endif
  for (i = 0; i < rows; ++i) {
    for (j = 0; j < cols; ++j) {
      out[(size_t)j * (size_t)rows + (size_t)i] = conj(a[(size_t)i * (size_t)cols + (size_t)j]);
    }
  }
}

static void wtec_mat_mul(const double complex *a, int m, int k, const double complex *b, int n, double complex *c) {
  int i, j, t;
  memset(c, 0, (size_t)m * (size_t)n * sizeof(double complex));
#if defined(_OPENMP)
#pragma omp parallel for private(j, t) schedule(static) if ((m * n * k) >= 32768)
#endif
  for (i = 0; i < m; ++i) {
    for (t = 0; t < k; ++t) {
      double complex av = a[(size_t)i * (size_t)k + (size_t)t];
      if (cabs(av) < WTEC_RGF_JSON_EPS) {
        continue;
      }
      for (j = 0; j < n; ++j) {
        c[(size_t)i * (size_t)n + (size_t)j] += av * b[(size_t)t * (size_t)n + (size_t)j];
      }
    }
  }
}

static void wtec_mat_add_inplace(double complex *dst, const double complex *src, int n) {
  int i;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (n >= 4096)
#endif
  for (i = 0; i < n; ++i) {
    dst[i] += src[i];
  }
}

static double wtec_mat_norm_fro(const double complex *a, int n) {
  int i;
  double sum = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for reduction(+:sum) schedule(static) if (n >= 4096)
#endif
  for (i = 0; i < n; ++i) {
    double v = cabs(a[i]);
    sum += v * v;
  }
  return sqrt(sum);
}

static int wtec_mat_inverse(const double complex *a, int n, double complex *out) {
  int i, j, k, pivot_row;
  double complex *aug = wtec_mat_alloc(n, 2 * n);
#if defined(_OPENMP)
#pragma omp parallel for private(j) schedule(static) if (n >= 32)
#endif
  for (i = 0; i < n; ++i) {
    for (j = 0; j < n; ++j) {
      aug[(size_t)i * (size_t)(2 * n) + (size_t)j] = a[(size_t)i * (size_t)n + (size_t)j];
      aug[(size_t)i * (size_t)(2 * n) + (size_t)(j + n)] = (i == j) ? 1.0 : 0.0;
    }
  }
  for (i = 0; i < n; ++i) {
    double best = 0.0;
    pivot_row = i;
    for (j = i; j < n; ++j) {
      double cand = cabs(aug[(size_t)j * (size_t)(2 * n) + (size_t)i]);
      if (cand > best) {
        best = cand;
        pivot_row = j;
      }
    }
    if (best < 1.0e-14) {
      free(aug);
      return -1;
    }
    if (pivot_row != i) {
      for (k = 0; k < 2 * n; ++k) {
        double complex tmp = aug[(size_t)i * (size_t)(2 * n) + (size_t)k];
        aug[(size_t)i * (size_t)(2 * n) + (size_t)k] = aug[(size_t)pivot_row * (size_t)(2 * n) + (size_t)k];
        aug[(size_t)pivot_row * (size_t)(2 * n) + (size_t)k] = tmp;
      }
    }
    {
      double complex piv = aug[(size_t)i * (size_t)(2 * n) + (size_t)i];
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (n >= 32)
#endif
      for (k = 0; k < 2 * n; ++k) {
        aug[(size_t)i * (size_t)(2 * n) + (size_t)k] /= piv;
      }
    }
#if defined(_OPENMP)
#pragma omp parallel for private(k) schedule(static) if (n >= 32)
#endif
    for (j = 0; j < n; ++j) {
      if (j == i) {
        continue;
      }
      {
        double complex factor = aug[(size_t)j * (size_t)(2 * n) + (size_t)i];
        if (cabs(factor) < WTEC_RGF_JSON_EPS) {
          continue;
        }
        for (k = 0; k < 2 * n; ++k) {
          aug[(size_t)j * (size_t)(2 * n) + (size_t)k] -= factor * aug[(size_t)i * (size_t)(2 * n) + (size_t)k];
        }
      }
    }
  }
  for (i = 0; i < n; ++i) {
    for (j = 0; j < n; ++j) {
      out[(size_t)i * (size_t)n + (size_t)j] = aug[(size_t)i * (size_t)(2 * n) + (size_t)(j + n)];
    }
  }
  free(aug);
  return 0;
}

static void wtec_build_resolvent(const double complex *h, int n, double complex z, double complex *out) {
  int i, j;
#if defined(_OPENMP)
#pragma omp parallel for private(j) schedule(static) if (n >= 32)
#endif
  for (i = 0; i < n; ++i) {
    for (j = 0; j < n; ++j) {
      out[(size_t)i * (size_t)n + (size_t)j] = -h[(size_t)i * (size_t)n + (size_t)j];
    }
    out[(size_t)i * (size_t)n + (size_t)i] += z;
  }
}

static void wtec_build_block(const wtec_ky_model_t *ky, int width_left, int width_right, int shift_x, int nz, double complex *out) {
  int it, a, z1, z2, b;
  const int nw = ky->num_wann;
  const int rows = width_left * nz * nw;
  const int cols = width_right * nz * nw;
  memset(out, 0, (size_t)rows * (size_t)cols * sizeof(double complex));
  for (it = 0; it < ky->n_terms; ++it) {
    const int rx = ky->rx[it];
    const int rz = ky->rz[it];
    const double complex *mat = ky->mat + (size_t)it * (size_t)nw * (size_t)nw;
    for (a = 0; a < width_left; ++a) {
      b = a + rx - shift_x;
      if (b < 0 || b >= width_right) {
        continue;
      }
      for (z1 = 0; z1 < nz; ++z1) {
        z2 = z1 + rz;
        if (z2 < 0 || z2 >= nz) {
          continue;
        }
        {
          int o1, o2;
          const int row_base = (a * nz + z1) * nw;
          const int col_base = (b * nz + z2) * nw;
          for (o1 = 0; o1 < nw; ++o1) {
            for (o2 = 0; o2 < nw; ++o2) {
              out[(size_t)(row_base + o1) * (size_t)cols + (size_t)(col_base + o2)] +=
                  mat[(size_t)o1 * (size_t)nw + (size_t)o2];
            }
          }
        }
      }
    }
  }
}

static void wtec_build_block_full(
    const wtec_hr_model_t *model,
    int width_left,
    int width_right,
    int shift_x,
    int ny,
    int nz,
    double complex *out) {
  int it, a, b, y1, y2, z1, z2, o1, o2;
  const int nw = model->num_wann;
  const int rows = width_left * ny * nz * nw;
  const int cols = width_right * ny * nz * nw;
  memset(out, 0, (size_t)rows * (size_t)cols * sizeof(double complex));
  for (it = 0; it < model->n_r; ++it) {
    const int rx = model->rx[it];
    const int ry = model->ry[it];
    const int rz = model->rz[it];
    const double inv_deg = 1.0 / (double)((model->deg != NULL && model->deg[it] > 0) ? model->deg[it] : 1);
    const double complex *mat = model->mat + (size_t)it * (size_t)nw * (size_t)nw;
    if (abs(ry) > ny - 1 || abs(rz) > nz - 1) {
      continue;
    }
    for (a = 0; a < width_left; ++a) {
      b = a + rx - shift_x;
      if (b < 0 || b >= width_right) {
        continue;
      }
      for (y1 = 0; y1 < ny; ++y1) {
        y2 = y1 + ry;
        if (y2 < 0 || y2 >= ny) {
          continue;
        }
        for (z1 = 0; z1 < nz; ++z1) {
          z2 = z1 + rz;
          if (z2 < 0 || z2 >= nz) {
            continue;
          }
          {
            const int row_base = ((a * ny + y1) * nz + z1) * nw;
            const int col_base = ((b * ny + y2) * nz + z2) * nw;
            for (o1 = 0; o1 < nw; ++o1) {
              for (o2 = 0; o2 < nw; ++o2) {
                out[(size_t)(row_base + o1) * (size_t)cols + (size_t)(col_base + o2)] +=
                    inv_deg * mat[(size_t)o1 * (size_t)nw + (size_t)o2];
              }
            }
          }
        }
      }
    }
  }
}

static int wtec_plan_boundary_preserving_widths(int nx, int lead_width, int **widths_out, int *n_slices_out) {
  int full, rem, i, cursor;
  int *widths = NULL;
  if (nx <= 0 || lead_width <= 0 || widths_out == NULL || n_slices_out == NULL) {
    return -1;
  }
  if (nx <= lead_width) {
    widths = (int *)wtec_calloc(1u, sizeof(int));
    widths[0] = nx;
    *widths_out = widths;
    *n_slices_out = 1;
    return 0;
  }
  full = nx / lead_width;
  rem = nx % lead_width;
  if (rem == 0) {
    widths = (int *)wtec_calloc((size_t)full, sizeof(int));
    for (i = 0; i < full; ++i) widths[i] = lead_width;
    *widths_out = widths;
    *n_slices_out = full;
    return 0;
  }
  if (full < 2) {
    widths = (int *)wtec_calloc(1u, sizeof(int));
    widths[0] = nx;
    *widths_out = widths;
    *n_slices_out = 1;
    return 0;
  }
  widths = (int *)wtec_calloc((size_t)full, sizeof(int));
  cursor = 0;
  for (i = 0; i < full - 2; ++i) {
    widths[cursor++] = lead_width;
  }
  widths[cursor++] = lead_width + rem;
  widths[cursor++] = lead_width;
  *widths_out = widths;
  *n_slices_out = full;
  return 0;
}

static void wtec_apply_onsite_region(
    double complex *h_slice,
    int width_x,
    int x_offset,
    int device_x_offset,
    int ny,
    int nz,
    int nw,
    const wtec_payload_t *payload) {
  int x, y, z, orb;
  if (payload == NULL || !payload->onsite_has_region || h_slice == NULL) {
    return;
  }
  for (x = 0; x < width_x; ++x) {
    const int gx = x_offset + x - device_x_offset;
    if (gx < payload->onsite_x_start || gx >= payload->onsite_x_stop) {
      continue;
    }
    for (y = 0; y < ny; ++y) {
      if (y < payload->onsite_y_start || y >= payload->onsite_y_stop) {
        continue;
      }
      for (z = 0; z < nz; ++z) {
        size_t base;
        if (z < payload->onsite_z_start || z >= payload->onsite_z_stop) {
          continue;
        }
        base = (size_t)(((x * ny + y) * nz + z) * nw);
        for (orb = 0; orb < nw; ++orb) {
          const size_t idx = base + (size_t)orb;
          h_slice[idx * (size_t)(width_x * ny * nz * nw) + idx] += payload->onsite_shift_ev;
        }
      }
    }
  }
}

static void wtec_apply_anderson_disorder(
    double complex *h_slice,
    int width_x,
    int x_offset,
    int device_x_offset,
    int nx_device,
    int ny,
    int nz,
    int nw,
    double disorder_strength,
    int seed) {
  int x, y, z, orb;
  if (h_slice == NULL || fabs(disorder_strength) <= WTEC_RGF_JSON_EPS) {
    return;
  }
  for (x = 0; x < width_x; ++x) {
    const int gx = x_offset + x - device_x_offset;
    size_t stride;
    if (gx < 0 || gx >= nx_device) {
      continue;
    }
    stride = (size_t)(width_x * ny * nz * nw);
    for (y = 0; y < ny; ++y) {
      for (z = 0; z < nz; ++z) {
        const size_t base = (size_t)(((x * ny + y) * nz + z) * nw);
        for (orb = 0; orb < nw; ++orb) {
          const size_t idx = base + (size_t)orb;
          const double xi = wtec_disorder_sample(seed, gx, y, z, orb);
          h_slice[idx * stride + idx] += disorder_strength * xi;
        }
      }
    }
  }
}

static int wtec_surface_green_sancho(
    const double complex *h0,
    const double complex *v,
    int n,
    double complex z,
    double complex *g_out,
    wtec_progress_t *progress,
    const wtec_point_context_t *ctx,
    const char *phase) {
  int iter;
  double complex *alpha = wtec_mat_alloc(n, n);
  double complex *beta = wtec_mat_alloc(n, n);
  double complex *eps = wtec_mat_alloc(n, n);
  double complex *eps_s = wtec_mat_alloc(n, n);
  double complex *m = wtec_mat_alloc(n, n);
  double complex *ginv = wtec_mat_alloc(n, n);
  double complex *term1 = wtec_mat_alloc(n, n);
  double complex *term2 = wtec_mat_alloc(n, n);
  double complex *next_a = wtec_mat_alloc(n, n);
  double complex *next_b = wtec_mat_alloc(n, n);
  int converged = 0;
  memcpy(alpha, v, (size_t)n * (size_t)n * sizeof(double complex));
  wtec_mat_conj_transpose(v, n, n, beta);
  memcpy(eps, h0, (size_t)n * (size_t)n * sizeof(double complex));
  memcpy(eps_s, h0, (size_t)n * (size_t)n * sizeof(double complex));
  wtec_progress_step(progress, ctx, phase, 0, WTEC_RGF_SANCHO_MAX_ITER);
  for (iter = 0; iter < WTEC_RGF_SANCHO_MAX_ITER; ++iter) {
    wtec_build_resolvent(eps, n, z, m);
    if (wtec_mat_inverse(m, n, ginv) != 0) {
      goto fail;
    }
    wtec_mat_mul(alpha, n, n, ginv, n, term1);
    wtec_mat_mul(term1, n, n, beta, n, next_a);   /* alpha g beta */
    wtec_mat_mul(beta, n, n, ginv, n, term2);
    wtec_mat_mul(term2, n, n, alpha, n, next_b);  /* beta g alpha */
    wtec_mat_add_inplace(eps_s, next_a, n * n);
    wtec_mat_add_inplace(eps, next_a, n * n);
    wtec_mat_add_inplace(eps, next_b, n * n);
    wtec_mat_mul(alpha, n, n, ginv, n, term1);
    wtec_mat_mul(term1, n, n, alpha, n, next_a);
    wtec_mat_mul(beta, n, n, ginv, n, term2);
    wtec_mat_mul(term2, n, n, beta, n, next_b);
    memcpy(alpha, next_a, (size_t)n * (size_t)n * sizeof(double complex));
    memcpy(beta, next_b, (size_t)n * (size_t)n * sizeof(double complex));
    if ((iter == 0) || (((iter + 1) % 4) == 0)) {
      wtec_progress_step(progress, ctx, phase, iter + 1, WTEC_RGF_SANCHO_MAX_ITER);
    }
    if (wtec_mat_norm_fro(alpha, n * n) + wtec_mat_norm_fro(beta, n * n) < WTEC_RGF_SANCHO_TOL) {
      converged = 1;
      wtec_progress_step(progress, ctx, phase, iter + 1, WTEC_RGF_SANCHO_MAX_ITER);
      break;
    }
  }
  wtec_build_resolvent(eps_s, n, z, m);
  if (wtec_mat_inverse(m, n, g_out) != 0) {
    goto fail;
  }
  if (!converged) {
    wtec_progress_step(progress, ctx, phase, WTEC_RGF_SANCHO_MAX_ITER, WTEC_RGF_SANCHO_MAX_ITER);
  }
  free(alpha); free(beta); free(eps); free(eps_s); free(m); free(ginv);
  free(term1); free(term2); free(next_a); free(next_b);
  return 0;
fail:
  free(alpha); free(beta); free(eps); free(eps_s); free(m); free(ginv);
  free(term1); free(term2); free(next_a); free(next_b);
  return -1;
}

static double wtec_trace_real(const double complex *a, int n) {
  int i;
  double out = 0.0;
  for (i = 0; i < n; ++i) {
    out += creal(a[(size_t)i * (size_t)n + (size_t)i]);
  }
  return out;
}

static void wtec_build_broadening(const double complex *sigma, int n, double complex *gamma) {
  int i, j;
  for (i = 0; i < n; ++i) {
    for (j = 0; j < n; ++j) {
      gamma[(size_t)i * (size_t)n + (size_t)j] =
          I * (sigma[(size_t)i * (size_t)n + (size_t)j] -
               conj(sigma[(size_t)j * (size_t)n + (size_t)i]));
    }
  }
}

static int wtec_read_complex_file(const char *path, double complex *out, size_t count) {
  FILE *fh = fopen(path, "rb");
  if (fh == NULL) {
    return -1;
  }
  if (fread(out, sizeof(double complex), count, fh) != count) {
    fclose(fh);
    return -1;
  }
  fclose(fh);
  return 0;
}

static int wtec_compute_transmission_from_blocks(
    const int *slice_dims,
    int n_slices,
    const double complex *h_blob,
    const double complex *v_blob,
    const double complex *sigma_l,
    const double complex *sigma_r,
    double energy,
    double eta,
    wtec_point_result_t *result) {
  int i;
  double complex z = energy + I * eta;
  double complex **h_slices = NULL;
  double complex **v_slices = NULL;
  double complex **g_left = NULL;
  double complex *gamma_l = NULL;
  double complex *gamma_r = NULL;
  double complex *g_nn = NULL;
  double complex *x = NULL;
  size_t h_off = 0;
  size_t v_off = 0;
  int ok = 0;
  const int d_first = slice_dims[0];
  const int d_last = slice_dims[n_slices - 1];

  if (n_slices <= 0 || d_first <= 0 || d_last <= 0) {
    return -1;
  }

  h_slices = (double complex **)wtec_calloc((size_t)n_slices, sizeof(double complex *));
  if (n_slices > 1) {
    v_slices = (double complex **)wtec_calloc((size_t)(n_slices - 1), sizeof(double complex *));
    g_left = (double complex **)wtec_calloc((size_t)(n_slices - 1), sizeof(double complex *));
  }

  for (i = 0; i < n_slices; ++i) {
    int d_i = slice_dims[i];
    size_t block_sz = (size_t)d_i * (size_t)d_i;
    h_slices[i] = wtec_mat_alloc(d_i, d_i);
    memcpy(h_slices[i], h_blob + h_off, block_sz * sizeof(double complex));
    h_off += block_sz;
    if (i < n_slices - 1) {
      int d_j = slice_dims[i + 1];
      size_t v_sz = (size_t)d_i * (size_t)d_j;
      v_slices[i] = wtec_mat_alloc(d_i, d_j);
      memcpy(v_slices[i], v_blob + v_off, v_sz * sizeof(double complex));
      v_off += v_sz;
    }
  }

  gamma_l = wtec_mat_alloc(d_first, d_first);
  gamma_r = wtec_mat_alloc(d_last, d_last);
  wtec_build_broadening(sigma_l, d_first, gamma_l);
  wtec_build_broadening(sigma_r, d_last, gamma_r);

  {
    double complex *m = wtec_mat_alloc(d_first, d_first);
    double complex *inv = wtec_mat_alloc(d_first, d_first);
    wtec_build_resolvent(h_slices[0], d_first, z, m);
    for (i = 0; i < d_first * d_first; ++i) {
      m[i] -= sigma_l[i];
    }
    if (n_slices == 1) {
      for (i = 0; i < d_first * d_first; ++i) {
        m[i] -= sigma_r[i];
      }
      if (wtec_mat_inverse(m, d_first, inv) != 0) {
        free(m);
        free(inv);
        goto cleanup;
      }
      g_nn = inv;
      free(m);
    } else {
      if (wtec_mat_inverse(m, d_first, inv) != 0) {
        free(m);
        free(inv);
        goto cleanup;
      }
      g_left[0] = inv;
      free(m);
    }
  }

  if (n_slices > 1) {
    for (i = 1; i < n_slices - 1; ++i) {
      const int d_prev = slice_dims[i - 1];
      const int d_i = slice_dims[i];
      double complex *v_h = wtec_mat_alloc(d_i, d_prev);
      double complex *term = wtec_mat_alloc(d_i, d_prev);
      double complex *sigma = wtec_mat_alloc(d_i, d_i);
      double complex *m = wtec_mat_alloc(d_i, d_i);
      wtec_mat_conj_transpose(v_slices[i - 1], d_prev, d_i, v_h);
      wtec_mat_mul(v_h, d_i, d_prev, g_left[i - 1], d_prev, term);
      wtec_mat_mul(term, d_i, d_prev, v_slices[i - 1], d_i, sigma);
      wtec_build_resolvent(h_slices[i], d_i, z, m);
      {
        int idx;
        for (idx = 0; idx < d_i * d_i; ++idx) {
          m[idx] -= sigma[idx];
        }
      }
      g_left[i] = wtec_mat_alloc(d_i, d_i);
      if (wtec_mat_inverse(m, d_i, g_left[i]) != 0) {
        free(v_h); free(term); free(sigma); free(m);
        goto cleanup;
      }
      free(v_h); free(term); free(sigma); free(m);
    }

    {
      const int d_prev = slice_dims[n_slices - 2];
      double complex *v_h = wtec_mat_alloc(d_last, d_prev);
      double complex *term = wtec_mat_alloc(d_last, d_prev);
      double complex *sigma = wtec_mat_alloc(d_last, d_last);
      double complex *m = wtec_mat_alloc(d_last, d_last);
      wtec_mat_conj_transpose(v_slices[n_slices - 2], d_prev, d_last, v_h);
      wtec_mat_mul(v_h, d_last, d_prev, g_left[n_slices - 2], d_prev, term);
      wtec_mat_mul(term, d_last, d_prev, v_slices[n_slices - 2], d_last, sigma);
      wtec_build_resolvent(h_slices[n_slices - 1], d_last, z, m);
      for (i = 0; i < d_last * d_last; ++i) {
        m[i] -= sigma[i];
        m[i] -= sigma_r[i];
      }
      g_nn = wtec_mat_alloc(d_last, d_last);
      if (wtec_mat_inverse(m, d_last, g_nn) != 0) {
        free(v_h); free(term); free(sigma); free(m);
        goto cleanup;
      }
      free(v_h); free(term); free(sigma); free(m);
    }
  }

  x = wtec_mat_alloc(d_last, d_last);
  memcpy(x, g_nn, (size_t)d_last * (size_t)d_last * sizeof(double complex));
  for (i = n_slices - 2; i >= 0; --i) {
    const int d_i = slice_dims[i];
    const int d_next = slice_dims[i + 1];
    double complex *term = wtec_mat_alloc(d_i, d_last);
    double complex *next = wtec_mat_alloc(d_i, d_last);
    wtec_mat_mul(v_slices[i], d_i, d_next, x, d_last, term);
    wtec_mat_mul(g_left[i], d_i, d_i, term, d_last, next);
    free(term);
    free(x);
    x = next;
  }

  {
    double complex *tmp_a = wtec_mat_alloc(d_first, d_last);
    double complex *x_h = wtec_mat_alloc(d_last, d_first);
    double complex *tmp_b = wtec_mat_alloc(d_first, d_first);
    double complex *tmp2 = wtec_mat_alloc(d_first, d_last);
    wtec_mat_mul(gamma_l, d_first, d_first, x, d_last, tmp_a);
    wtec_mat_conj_transpose(x, d_first, d_last, x_h);
    wtec_mat_mul(tmp_a, d_first, d_last, gamma_r, d_last, tmp2);
    wtec_mat_mul(tmp2, d_first, d_last, x_h, d_first, tmp_b);
    result->transmission = wtec_trace_real(tmp_b, d_first);
    free(tmp_a); free(x_h); free(tmp_b); free(tmp2);
  }

  result->p_eff = 0;
  result->slice_count = n_slices;
  result->superslice_dim = d_first;
  ok = 1;

cleanup:
  if (h_slices != NULL) {
    for (i = 0; i < n_slices; ++i) {
      free(h_slices[i]);
    }
  }
  if (v_slices != NULL) {
    for (i = 0; i < n_slices - 1; ++i) {
      free(v_slices[i]);
    }
  }
  if (g_left != NULL) {
    for (i = 0; i < n_slices - 1; ++i) {
      free(g_left[i]);
    }
  }
  free(h_slices); free(v_slices); free(g_left);
  free(gamma_l); free(gamma_r); free(g_nn); free(x);
  return ok ? 0 : -1;
}

static int wtec_compute_transmission_block_validation(const wtec_payload_t *payload, wtec_point_result_t *result) {
  int i;
  int n_slices = payload->n_slice_dims;
  size_t total_h = 0;
  size_t total_v = 0;
  double complex *h_blob = NULL;
  double complex *v_blob = NULL;
  double complex *sigma_l = NULL;
  double complex *sigma_r = NULL;
  int d_first, d_last;
  for (i = 0; i < n_slices; ++i) {
    total_h += (size_t)payload->slice_dims[i] * (size_t)payload->slice_dims[i];
    if (i < n_slices - 1) {
      total_v += (size_t)payload->slice_dims[i] * (size_t)payload->slice_dims[i + 1];
    }
  }
  d_first = payload->slice_dims[0];
  d_last = payload->slice_dims[n_slices - 1];
  h_blob = wtec_mat_alloc((int)total_h, 1);
  v_blob = wtec_mat_alloc((int)(total_v > 0 ? total_v : 1), 1);
  sigma_l = wtec_mat_alloc(d_first, d_first);
  sigma_r = wtec_mat_alloc(d_last, d_last);
  if (wtec_read_complex_file(payload->h_slices_path, h_blob, total_h) != 0 ||
      (total_v > 0 && wtec_read_complex_file(payload->v_slices_path, v_blob, total_v) != 0) ||
      wtec_read_complex_file(payload->sigma_left_path, sigma_l, (size_t)d_first * (size_t)d_first) != 0 ||
      wtec_read_complex_file(payload->sigma_right_path, sigma_r, (size_t)d_last * (size_t)d_last) != 0) {
    free(h_blob); free(v_blob); free(sigma_l); free(sigma_r);
    return -1;
  }
  i = wtec_compute_transmission_from_blocks(
      payload->slice_dims,
      n_slices,
      h_blob,
      v_blob,
      sigma_l,
      sigma_r,
      payload->energy,
      payload->eta,
      result);
  free(h_blob); free(v_blob); free(sigma_l); free(sigma_r);
  return i;
}

static int wtec_try_load_sigma_override(
    const wtec_payload_t *payload,
    int d_left,
    int d_right,
    double complex *sigma_l,
    double complex *sigma_r,
    double complex *gamma_l,
    double complex *gamma_r) {
  if (payload == NULL || sigma_l == NULL || sigma_r == NULL ||
      gamma_l == NULL || gamma_r == NULL) {
    return -1;
  }
  if (payload->sigma_left_path[0] == '\0' || payload->sigma_right_path[0] == '\0') {
    return 1;
  }
  if (wtec_read_complex_file(payload->sigma_left_path, sigma_l, (size_t)d_left * (size_t)d_left) != 0 ||
      wtec_read_complex_file(payload->sigma_right_path, sigma_r, (size_t)d_right * (size_t)d_right) != 0) {
    return -1;
  }
  wtec_build_broadening(sigma_l, d_left, gamma_l);
  wtec_build_broadening(sigma_r, d_right, gamma_r);
  return 0;
}

static void wtec_reverse_lead_rows(
    const double complex *src,
    int rows,
    int cols,
    int p_eff,
    int cross_block,
    double complex *dst) {
  int x, t, c;
  memset(dst, 0, (size_t)rows * (size_t)cols * sizeof(double complex));
  if (rows != p_eff * cross_block) {
    memcpy(dst, src, (size_t)rows * (size_t)cols * sizeof(double complex));
    return;
  }
  for (x = 0; x < p_eff; ++x) {
    for (t = 0; t < cross_block; ++t) {
        const int src_row = x * cross_block + t;
        const int dst_row = (p_eff - 1 - x) * cross_block + t;
        for (c = 0; c < cols; ++c) {
          dst[(size_t)dst_row * (size_t)cols + (size_t)c] =
              src[(size_t)src_row * (size_t)cols + (size_t)c];
        }
    }
  }
}

static void wtec_reverse_lead_cols(
    const double complex *src,
    int rows,
    int cols,
    int p_eff,
    int cross_block,
    double complex *dst) {
  int r, x, t;
  memset(dst, 0, (size_t)rows * (size_t)cols * sizeof(double complex));
  if (cols != p_eff * cross_block) {
    memcpy(dst, src, (size_t)rows * (size_t)cols * sizeof(double complex));
    return;
  }
  for (r = 0; r < rows; ++r) {
    for (x = 0; x < p_eff; ++x) {
      for (t = 0; t < cross_block; ++t) {
          const int src_col = x * cross_block + t;
          const int dst_col = (p_eff - 1 - x) * cross_block + t;
          dst[(size_t)r * (size_t)cols + (size_t)dst_col] =
              src[(size_t)r * (size_t)cols + (size_t)src_col];
      }
    }
  }
}

static int wtec_compute_transmission_periodic_y(
    const wtec_hr_model_t *model,
    double ky_frac,
    int nx,
    int nz,
    double energy,
    double eta,
    wtec_progress_t *progress,
    const wtec_point_context_t *ctx,
    wtec_point_result_t *result) {
  wtec_ky_model_t ky;
  int i, n_slices, width_last, p_eff;
  int *widths = NULL;
  double complex z = energy + I * eta;
  if (wtec_build_ky_model(model, ky_frac, &ky) != 0) {
    return -1;
  }
  wtec_progress_step(progress, ctx, "periodic_y_build_ky_model", -1, -1);
  p_eff = wtec_p_eff_for_model_nz(model, nz);
  if (p_eff <= 0) {
    wtec_free_ky_model(&ky);
    return -1;
  }
  n_slices = (nx + p_eff - 1) / p_eff;
  width_last = nx - p_eff * (n_slices - 1);
  widths = (int *)wtec_calloc((size_t)n_slices, sizeof(int));
  for (i = 0; i < n_slices; ++i) {
    widths[i] = p_eff;
  }
  if (n_slices > 0 && width_last > 0 && width_last < p_eff) {
    widths[n_slices - 1] = width_last;
  }
  {
    const int nw = model->num_wann;
    const int d_lead = p_eff * nz * nw;
    const int d_last = widths[n_slices - 1] * nz * nw;
    double complex *h_lead = wtec_mat_alloc(d_lead, d_lead);
    double complex *v_lead = wtec_mat_alloc(d_lead, d_lead);
    double complex *v_lead_r = wtec_mat_alloc(d_lead, d_lead);
    double complex *g_surf = wtec_mat_alloc(d_lead, d_lead);
    double complex *g_surf_r = wtec_mat_alloc(d_lead, d_lead);
    double complex *sigma_l = wtec_mat_alloc(d_lead, d_lead);
    double complex *gamma_l = wtec_mat_alloc(d_lead, d_lead);
    double complex *c_left = wtec_mat_alloc(d_lead, d_lead);
    double complex *c_left_h = wtec_mat_alloc(d_lead, d_lead);
    double complex *tmp = NULL;
    double complex *tmp2 = NULL;
    double complex **g_left = NULL;
    double complex **v_slices = NULL;
    double complex **h_slices = NULL;
    double complex *sigma_r = NULL;
    double complex *gamma_r = NULL;
    double complex *c_right = NULL;
    double complex *c_right_perm = NULL;
    double complex *c_right_h = NULL;
    double complex *g_nn = NULL;
    double complex *x = NULL;
    int ok = 0;

    wtec_build_block(&ky, p_eff, p_eff, 0, nz, h_lead);
    wtec_build_block(&ky, p_eff, p_eff, p_eff, nz, v_lead);
    wtec_progress_step(progress, ctx, "periodic_y_build_leads", -1, -1);
    wtec_mat_conj_transpose(v_lead, d_lead, d_lead, v_lead_r);
    if (wtec_surface_green_sancho(
            h_lead,
            v_lead,
            d_lead,
            z,
            g_surf,
            progress,
            ctx,
            "periodic_y_surface_green_left") != 0) {
      goto numeric_cleanup;
    }
    if (wtec_surface_green_sancho(
            h_lead,
            v_lead_r,
            d_lead,
            z,
            g_surf_r,
            progress,
            ctx,
            "periodic_y_surface_green_right") != 0) {
      goto numeric_cleanup;
    }
    wtec_progress_step(progress, ctx, "periodic_y_surface_green", -1, -1);

    h_slices = (double complex **)wtec_calloc((size_t)n_slices, sizeof(double complex *));
    if (n_slices > 1) {
      v_slices = (double complex **)wtec_calloc((size_t)(n_slices - 1), sizeof(double complex *));
      g_left = (double complex **)wtec_calloc((size_t)(n_slices - 1), sizeof(double complex *));
    }
    for (i = 0; i < n_slices; ++i) {
      const int d_i = widths[i] * nz * nw;
      h_slices[i] = wtec_mat_alloc(d_i, d_i);
      wtec_build_block(&ky, widths[i], widths[i], 0, nz, h_slices[i]);
      if (i < n_slices - 1) {
        const int d_j = widths[i + 1] * nz * nw;
        v_slices[i] = wtec_mat_alloc(d_i, d_j);
        wtec_build_block(&ky, widths[i], widths[i + 1], widths[i], nz, v_slices[i]);
      }
      wtec_progress_step(progress, ctx, "periodic_y_build_slices", i + 1, n_slices);
    }

    /* Left self-energy on the first slice. */
    wtec_reverse_lead_rows(v_lead, d_lead, d_lead, p_eff, nz * nw, c_left);
    tmp = wtec_mat_alloc(d_lead, d_lead);
    wtec_mat_mul(c_left, d_lead, d_lead, g_surf, d_lead, tmp);
    wtec_mat_conj_transpose(c_left, d_lead, d_lead, c_left_h);
    wtec_mat_mul(tmp, d_lead, d_lead, c_left_h, d_lead, sigma_l);
    wtec_build_broadening(sigma_l, d_lead, gamma_l);
    wtec_progress_step(progress, ctx, "periodic_y_left_sigma", -1, -1);

    /* Right self-energy on the last slice. */
    sigma_r = wtec_mat_alloc(d_last, d_last);
    gamma_r = wtec_mat_alloc(d_last, d_last);
    c_right = wtec_mat_alloc(d_last, d_lead);
    c_right_perm = wtec_mat_alloc(d_last, d_lead);
    c_right_h = wtec_mat_alloc(d_lead, d_last);
    wtec_build_block(&ky, widths[n_slices - 1], p_eff, widths[n_slices - 1], nz, c_right);
    wtec_reverse_lead_cols(c_right, d_last, d_lead, p_eff, nz * nw, c_right_perm);
    wtec_mat_conj_transpose(c_right_perm, d_last, d_lead, c_right_h);
    tmp2 = wtec_mat_alloc(d_last, d_lead);
    wtec_mat_mul(c_right_perm, d_last, d_lead, g_surf_r, d_lead, tmp2);
    wtec_mat_mul(tmp2, d_last, d_lead, c_right_h, d_last, sigma_r);
    wtec_build_broadening(sigma_r, d_last, gamma_r);
    wtec_progress_step(progress, ctx, "periodic_y_right_sigma", -1, -1);

    {
      int d0 = d_lead;
      double complex *m = wtec_mat_alloc(d0, d0);
      double complex *inv = wtec_mat_alloc(d0, d0);
      wtec_build_resolvent(h_slices[0], d0, z, m);
      for (i = 0; i < d0 * d0; ++i) {
        m[i] -= sigma_l[i];
      }
      if (n_slices == 1) {
        for (i = 0; i < d0 * d0; ++i) {
          m[i] -= sigma_r[i];
        }
        if (wtec_mat_inverse(m, d0, inv) != 0) {
          free(m);
          free(inv);
          goto numeric_cleanup;
        }
        g_nn = inv;
        free(m);
      } else {
        if (wtec_mat_inverse(m, d0, inv) != 0) {
          free(m);
          free(inv);
          goto numeric_cleanup;
        }
        g_left[0] = inv;
        free(m);
      }
    }
    wtec_progress_step(progress, ctx, "periodic_y_initial_inverse", -1, -1);

    if (n_slices > 1) {
      for (i = 1; i < n_slices - 1; ++i) {
        const int d_prev = widths[i - 1] * nz * nw;
        const int d_i = widths[i] * nz * nw;
        double complex *v_h = wtec_mat_alloc(d_i, d_prev);
        double complex *term = wtec_mat_alloc(d_i, d_prev);
        double complex *sigma = wtec_mat_alloc(d_i, d_i);
        double complex *m = wtec_mat_alloc(d_i, d_i);
        wtec_mat_conj_transpose(v_slices[i - 1], d_prev, d_i, v_h);
        wtec_mat_mul(v_h, d_i, d_prev, g_left[i - 1], d_prev, term);
        wtec_mat_mul(term, d_i, d_prev, v_slices[i - 1], d_i, sigma);
        wtec_build_resolvent(h_slices[i], d_i, z, m);
        {
          int idx;
          for (idx = 0; idx < d_i * d_i; ++idx) {
            m[idx] -= sigma[idx];
          }
        }
        g_left[i] = wtec_mat_alloc(d_i, d_i);
        if (wtec_mat_inverse(m, d_i, g_left[i]) != 0) {
          free(v_h); free(term); free(sigma); free(m);
          goto numeric_cleanup;
        }
        free(v_h); free(term); free(sigma); free(m);
        wtec_progress_step(progress, ctx, "periodic_y_forward_sweep", i, n_slices);
      }

      {
        const int d_prev = widths[n_slices - 2] * nz * nw;
        double complex *v_h = wtec_mat_alloc(d_last, d_prev);
        double complex *term = wtec_mat_alloc(d_last, d_prev);
        double complex *sigma = wtec_mat_alloc(d_last, d_last);
        double complex *m = wtec_mat_alloc(d_last, d_last);
        wtec_mat_conj_transpose(v_slices[n_slices - 2], d_prev, d_last, v_h);
        wtec_mat_mul(v_h, d_last, d_prev, g_left[n_slices - 2], d_prev, term);
        wtec_mat_mul(term, d_last, d_prev, v_slices[n_slices - 2], d_last, sigma);
        wtec_build_resolvent(h_slices[n_slices - 1], d_last, z, m);
        for (i = 0; i < d_last * d_last; ++i) {
          m[i] -= sigma[i];
          m[i] -= sigma_r[i];
        }
        g_nn = wtec_mat_alloc(d_last, d_last);
        if (wtec_mat_inverse(m, d_last, g_nn) != 0) {
          free(v_h); free(term); free(sigma); free(m);
          goto numeric_cleanup;
        }
        free(v_h); free(term); free(sigma); free(m);
        wtec_progress_step(progress, ctx, "periodic_y_last_inverse", n_slices - 1, n_slices);
      }
    }

    x = wtec_mat_alloc(d_last, d_last);
    memcpy(x, g_nn, (size_t)d_last * (size_t)d_last * sizeof(double complex));
    for (i = n_slices - 2; i >= 0; --i) {
      const int d_i = widths[i] * nz * nw;
      const int d_next = widths[i + 1] * nz * nw;
      double complex *term = wtec_mat_alloc(d_i, d_last);
      double complex *next = wtec_mat_alloc(d_i, d_last);
      wtec_mat_mul(v_slices[i], d_i, d_next, x, d_last, term);
      wtec_mat_mul(g_left[i], d_i, d_i, term, d_last, next);
      free(term);
      free(x);
      x = next;
      wtec_progress_step(progress, ctx, "periodic_y_backward_sweep", i + 1, n_slices);
    }

    {
      double complex *tmp_a = wtec_mat_alloc(d_lead, d_last);
      double complex *x_h = wtec_mat_alloc(d_last, d_lead);
      double complex *tmp_b = wtec_mat_alloc(d_lead, d_lead);
      if (tmp2 == NULL) {
        tmp2 = wtec_mat_alloc(d_lead, d_last);
      }
      wtec_mat_mul(gamma_l, d_lead, d_lead, x, d_last, tmp_a);
      wtec_mat_conj_transpose(x, d_lead, d_last, x_h);
      wtec_mat_mul(tmp_a, d_lead, d_last, gamma_r, d_last, tmp2);
      wtec_mat_mul(tmp2, d_lead, d_last, x_h, d_lead, tmp_b);
      result->transmission = wtec_trace_real(tmp_b, d_lead);
      free(tmp_a); free(x_h); free(tmp_b);
    }
    wtec_progress_step(progress, ctx, "periodic_y_fisher_lee", -1, -1);

    result->p_eff = p_eff;
    result->slice_count = n_slices;
    result->superslice_dim = d_lead;
    ok = 1;

numeric_cleanup:
    if (h_slices != NULL) {
      for (i = 0; i < n_slices; ++i) {
        free(h_slices[i]);
      }
    }
    if (v_slices != NULL) {
      for (i = 0; i < n_slices - 1; ++i) {
        free(v_slices[i]);
      }
    }
    if (g_left != NULL) {
      for (i = 0; i < n_slices - 1; ++i) {
        free(g_left[i]);
      }
    }
    free(h_slices); free(v_slices); free(g_left);
    free(h_lead); free(v_lead); free(v_lead_r); free(g_surf); free(g_surf_r); free(sigma_l); free(gamma_l); free(c_left); free(c_left_h);
    free(tmp); free(tmp2); free(sigma_r); free(gamma_r); free(c_right); free(c_right_perm); free(c_right_h); free(g_nn); free(x);
    free(widths);
    wtec_free_ky_model(&ky);
    return ok ? 0 : -1;
  }
}

static int wtec_compute_transmission_full_finite(
    const wtec_hr_model_t *model,
    const wtec_payload_t *payload,
    int nx,
    int ny,
    int nz,
    double energy,
    double eta,
    wtec_progress_t *progress,
    const wtec_point_context_t *ctx,
    wtec_point_result_t *result) {
  int i, n_slices, p_eff;
  int *widths = NULL;
  double complex z = energy + I * eta;
  int max_ry = 0, max_rz = 0;
  int pad_x = 0;
  int nx_effective;
  p_eff = wtec_p_eff_for_model_full(model, ny, nz);
  if (p_eff <= 0) {
    return -1;
  }
  wtec_active_transverse_spans_full(model, ny, nz, &max_ry, &max_rz);
  pad_x = (p_eff > 0 ? (p_eff - 1) : 0) + max_ry + max_rz;
  nx_effective = nx + 2 * pad_x;
  if (wtec_plan_boundary_preserving_widths(nx_effective, p_eff, &widths, &n_slices) != 0 || n_slices <= 0) {
    return -1;
  }
  wtec_progress_step(progress, ctx, "full_finite_plan", -1, -1);

  {
    const int nw = model->num_wann;
    const int cross_block = ny * nz * nw;
    const int d_first = widths[0] * cross_block;
    const int d_lead = p_eff * cross_block;
    const int d_last = widths[n_slices - 1] * cross_block;
    double complex *h_lead = wtec_mat_alloc(d_lead, d_lead);
    double complex *v_lead = wtec_mat_alloc(d_lead, d_lead);
    double complex *v_lead_r = wtec_mat_alloc(d_lead, d_lead);
    double complex *g_surf = wtec_mat_alloc(d_lead, d_lead);
    double complex *g_surf_r = wtec_mat_alloc(d_lead, d_lead);
    double complex *sigma_l = wtec_mat_alloc(d_first, d_first);
    double complex *gamma_l = wtec_mat_alloc(d_first, d_first);
    double complex *c_left = NULL;
    double complex *c_left_h = NULL;
    double complex *tmp = NULL;
    double complex *tmp2 = NULL;
    double complex **g_left = NULL;
    double complex **v_slices = NULL;
    double complex **h_slices = NULL;
    double complex *sigma_r = wtec_mat_alloc(d_last, d_last);
    double complex *gamma_r = wtec_mat_alloc(d_last, d_last);
    double complex *c_right = NULL;
    double complex *c_right_perm = NULL;
    double complex *c_right_h = NULL;
    double complex *g_nn = NULL;
    double complex *x = NULL;
    int ok = 0;
    int sigma_override_status = 1;
    const char *fail_stage = "not_set";

    h_slices = (double complex **)wtec_calloc((size_t)n_slices, sizeof(double complex *));
    if (n_slices > 1) {
      v_slices = (double complex **)wtec_calloc((size_t)(n_slices - 1), sizeof(double complex *));
      g_left = (double complex **)wtec_calloc((size_t)(n_slices - 1), sizeof(double complex *));
    }
    for (i = 0; i < n_slices; ++i) {
      const int d_i = widths[i] * cross_block;
      int x_offset = 0;
      int j;
      for (j = 0; j < i; ++j) {
        x_offset += widths[j];
      }
      h_slices[i] = wtec_mat_alloc(d_i, d_i);
      wtec_build_block_full(model, widths[i], widths[i], 0, ny, nz, h_slices[i]);
      if (ctx != NULL && fabs(ctx->disorder_strength) > WTEC_RGF_JSON_EPS) {
        wtec_apply_anderson_disorder(
            h_slices[i],
            widths[i],
            x_offset,
            pad_x,
            nx,
            ny,
            nz,
            nw,
            ctx->disorder_strength,
            ctx->seed);
      }
      wtec_apply_onsite_region(h_slices[i], widths[i], x_offset, pad_x, ny, nz, nw, payload);
      if (i < n_slices - 1) {
        const int d_j = widths[i + 1] * cross_block;
        v_slices[i] = wtec_mat_alloc(d_i, d_j);
        wtec_build_block_full(model, widths[i], widths[i + 1], widths[i], ny, nz, v_slices[i]);
      }
      wtec_progress_step(progress, ctx, "full_finite_build_slices", i + 1, n_slices);
    }

    sigma_override_status = wtec_try_load_sigma_override(
        payload,
        d_first,
        d_last,
        sigma_l,
        sigma_r,
        gamma_l,
        gamma_r);
    if (sigma_override_status < 0) {
      fail_stage = "load_sigma_override";
      goto numeric_cleanup;
    }
    wtec_progress_step(progress, ctx, "full_finite_sigma_override", -1, -1);
    if (sigma_override_status > 0) {
      wtec_build_block_full(model, p_eff, p_eff, 0, ny, nz, h_lead);
      wtec_build_block_full(model, p_eff, p_eff, p_eff, ny, nz, v_lead);
      wtec_progress_step(progress, ctx, "full_finite_build_leads", -1, -1);
      wtec_mat_conj_transpose(v_lead, d_lead, d_lead, v_lead_r);
      if (wtec_surface_green_sancho(
              h_lead,
              v_lead,
              d_lead,
              z,
              g_surf,
              progress,
              ctx,
              "full_finite_surface_green_left") != 0) {
        fail_stage = "sigma_left_surface_green";
        goto numeric_cleanup;
      }
      if (wtec_surface_green_sancho(
              h_lead,
              v_lead_r,
              d_lead,
              z,
              g_surf_r,
              progress,
              ctx,
              "full_finite_surface_green_right") != 0) {
        fail_stage = "sigma_right_surface_green";
        goto numeric_cleanup;
      }
      wtec_progress_step(progress, ctx, "full_finite_surface_green", -1, -1);

      c_left = wtec_mat_alloc(d_first, d_lead);
      c_left_h = wtec_mat_alloc(d_lead, d_first);
      wtec_build_block_full(model, widths[0], p_eff, p_eff, ny, nz, c_left);
      tmp = wtec_mat_alloc(d_first, d_lead);
      wtec_mat_mul(c_left, d_first, d_lead, g_surf, d_lead, tmp);
      wtec_mat_conj_transpose(c_left, d_first, d_lead, c_left_h);
      wtec_mat_mul(tmp, d_first, d_lead, c_left_h, d_first, sigma_l);
      wtec_build_broadening(sigma_l, d_first, gamma_l);
      wtec_progress_step(progress, ctx, "full_finite_left_sigma", -1, -1);

      c_right = wtec_mat_alloc(d_last, d_lead);
      c_right_perm = wtec_mat_alloc(d_last, d_lead);
      c_right_h = wtec_mat_alloc(d_lead, d_last);
      wtec_build_block_full(model, widths[n_slices - 1], p_eff, widths[n_slices - 1], ny, nz, c_right);
      wtec_reverse_lead_cols(c_right, d_last, d_lead, p_eff, cross_block, c_right_perm);
      wtec_mat_conj_transpose(c_right_perm, d_last, d_lead, c_right_h);
      tmp2 = wtec_mat_alloc(d_last, d_lead);
      wtec_mat_mul(c_right_perm, d_last, d_lead, g_surf_r, d_lead, tmp2);
      wtec_mat_mul(tmp2, d_last, d_lead, c_right_h, d_last, sigma_r);
      wtec_build_broadening(sigma_r, d_last, gamma_r);
      wtec_progress_step(progress, ctx, "full_finite_right_sigma", -1, -1);
    }

    {
      const int d0 = d_first;
      double complex *m = wtec_mat_alloc(d0, d0);
      double complex *inv = wtec_mat_alloc(d0, d0);
      wtec_build_resolvent(h_slices[0], d0, z, m);
      for (i = 0; i < d0 * d0; ++i) {
        m[i] -= sigma_l[i];
      }
      if (n_slices == 1) {
        for (i = 0; i < d0 * d0; ++i) {
          m[i] -= sigma_r[i];
        }
        if (wtec_mat_inverse(m, d0, inv) != 0) {
          fail_stage = "inv_first_single";
          free(m);
          free(inv);
          goto numeric_cleanup;
        }
        g_nn = inv;
        free(m);
      } else {
        if (wtec_mat_inverse(m, d0, inv) != 0) {
          fail_stage = "inv_first";
          free(m);
          free(inv);
          goto numeric_cleanup;
        }
        g_left[0] = inv;
        free(m);
      }
    }
    wtec_progress_step(progress, ctx, "full_finite_initial_inverse", -1, -1);

    if (n_slices > 1) {
      for (i = 1; i < n_slices - 1; ++i) {
        const int d_prev = widths[i - 1] * cross_block;
        const int d_i = widths[i] * cross_block;
        double complex *v_h = wtec_mat_alloc(d_i, d_prev);
        double complex *term = wtec_mat_alloc(d_i, d_prev);
        double complex *sigma = wtec_mat_alloc(d_i, d_i);
        double complex *m = wtec_mat_alloc(d_i, d_i);
        wtec_mat_conj_transpose(v_slices[i - 1], d_prev, d_i, v_h);
        wtec_mat_mul(v_h, d_i, d_prev, g_left[i - 1], d_prev, term);
        wtec_mat_mul(term, d_i, d_prev, v_slices[i - 1], d_i, sigma);
        wtec_build_resolvent(h_slices[i], d_i, z, m);
        {
          int idx;
          for (idx = 0; idx < d_i * d_i; ++idx) {
            m[idx] -= sigma[idx];
          }
        }
        g_left[i] = wtec_mat_alloc(d_i, d_i);
        if (wtec_mat_inverse(m, d_i, g_left[i]) != 0) {
          fail_stage = "inv_middle";
          free(v_h); free(term); free(sigma); free(m);
          goto numeric_cleanup;
        }
        free(v_h); free(term); free(sigma); free(m);
        wtec_progress_step(progress, ctx, "full_finite_forward_sweep", i, n_slices);
      }

      {
        const int d_prev = widths[n_slices - 2] * cross_block;
        double complex *v_h = wtec_mat_alloc(d_last, d_prev);
        double complex *term = wtec_mat_alloc(d_last, d_prev);
        double complex *sigma = wtec_mat_alloc(d_last, d_last);
        double complex *m = wtec_mat_alloc(d_last, d_last);
        wtec_mat_conj_transpose(v_slices[n_slices - 2], d_prev, d_last, v_h);
        wtec_mat_mul(v_h, d_last, d_prev, g_left[n_slices - 2], d_prev, term);
        wtec_mat_mul(term, d_last, d_prev, v_slices[n_slices - 2], d_last, sigma);
        wtec_build_resolvent(h_slices[n_slices - 1], d_last, z, m);
        for (i = 0; i < d_last * d_last; ++i) {
          m[i] -= sigma[i];
          m[i] -= sigma_r[i];
        }
        g_nn = wtec_mat_alloc(d_last, d_last);
        if (wtec_mat_inverse(m, d_last, g_nn) != 0) {
          fail_stage = "inv_last";
          free(v_h); free(term); free(sigma); free(m);
          goto numeric_cleanup;
        }
        free(v_h); free(term); free(sigma); free(m);
        wtec_progress_step(progress, ctx, "full_finite_last_inverse", n_slices - 1, n_slices);
      }
    }

    x = wtec_mat_alloc(d_last, d_last);
    memcpy(x, g_nn, (size_t)d_last * (size_t)d_last * sizeof(double complex));
    for (i = n_slices - 2; i >= 0; --i) {
      const int d_i = widths[i] * cross_block;
      const int d_next = widths[i + 1] * cross_block;
      double complex *term = wtec_mat_alloc(d_i, d_last);
      double complex *next = wtec_mat_alloc(d_i, d_last);
      wtec_mat_mul(v_slices[i], d_i, d_next, x, d_last, term);
      wtec_mat_mul(g_left[i], d_i, d_i, term, d_last, next);
      free(term);
      free(x);
      x = next;
      wtec_progress_step(progress, ctx, "full_finite_backward_sweep", i + 1, n_slices);
    }

    {
      double complex *tmp_a = wtec_mat_alloc(d_first, d_last);
      double complex *x_h = wtec_mat_alloc(d_last, d_first);
      double complex *tmp_b = wtec_mat_alloc(d_first, d_first);
      if (tmp2 == NULL) {
        tmp2 = wtec_mat_alloc(d_first, d_last);
      }
      wtec_mat_mul(gamma_l, d_first, d_first, x, d_last, tmp_a);
      wtec_mat_conj_transpose(x, d_first, d_last, x_h);
      wtec_mat_mul(tmp_a, d_first, d_last, gamma_r, d_last, tmp2);
      wtec_mat_mul(tmp2, d_first, d_last, x_h, d_first, tmp_b);
      result->transmission = wtec_trace_real(tmp_b, d_first);
      free(tmp_a); free(x_h); free(tmp_b);
    }
    wtec_progress_step(progress, ctx, "full_finite_fisher_lee", -1, -1);

    result->p_eff = p_eff;
    result->slice_count = n_slices;
    result->superslice_dim = (d_first > d_last) ? d_first : d_last;
    ok = 1;

numeric_cleanup:
    if (!ok) {
      fprintf(
          stderr,
          "[rgf][full_finite] failure stage=%s sigma_override=%d p_eff=%d n_slices=%d d_lead=%d d_last=%d nx=%d ny=%d nz=%d\n",
          fail_stage,
          sigma_override_status == 0 ? 1 : 0,
          p_eff,
          n_slices,
          d_first,
          d_last,
          nx,
          ny,
          nz);
    }
    if (h_slices != NULL) {
      for (i = 0; i < n_slices; ++i) {
        free(h_slices[i]);
      }
    }
    if (v_slices != NULL) {
      for (i = 0; i < n_slices - 1; ++i) {
        free(v_slices[i]);
      }
    }
    if (g_left != NULL) {
      for (i = 0; i < n_slices - 1; ++i) {
        free(g_left[i]);
      }
    }
    free(h_slices); free(v_slices); free(g_left);
    free(h_lead); free(v_lead); free(v_lead_r); free(g_surf); free(g_surf_r); free(sigma_l); free(gamma_l); free(c_left); free(c_left_h);
    free(tmp); free(tmp2); free(sigma_r); free(gamma_r); free(c_right); free(c_right_perm); free(c_right_h); free(g_nn); free(x);
    free(widths);
    return ok ? 0 : -1;
  }
}

static void wtec_write_int_array(FILE *fh, const int *vals, int n) {
  int i;
  fputc('[', fh);
  for (i = 0; i < n; ++i) {
    if (i > 0) {
      fputs(", ", fh);
    }
    fprintf(fh, "%d", vals[i]);
  }
  fputc(']', fh);
}

static void wtec_write_double_array(FILE *fh, const double *vals, int n) {
  int i;
  fputc('[', fh);
  for (i = 0; i < n; ++i) {
    if (i > 0) {
      fputs(", ", fh);
    }
    fprintf(fh, "%.16g", vals[i]);
  }
  fputc(']', fh);
}

static void wtec_write_double_matrix(FILE *fh, const double *vals, int rows, int cols) {
  int i;
  fputc('[', fh);
  for (i = 0; i < rows; ++i) {
    const double *row = NULL;
    if (i > 0) {
      fputs(", ", fh);
    }
    if (vals != NULL && cols > 0) {
      row = vals + (size_t)i * (size_t)cols;
    }
    wtec_write_double_array(fh, row, cols);
  }
  fputc(']', fh);
}

static void wtec_write_double_tensor3(FILE *fh, const double *vals, int dim0, int dim1, int dim2) {
  int i;
  fputc('[', fh);
  for (i = 0; i < dim0; ++i) {
    const double *slab = NULL;
    if (i > 0) {
      fputs(", ", fh);
    }
    if (vals != NULL && dim1 > 0 && dim2 > 0) {
      slab = vals + (size_t)i * (size_t)dim1 * (size_t)dim2;
    }
    wtec_write_double_matrix(
        fh,
        slab,
        dim1,
        dim2);
  }
  fputc(']', fh);
}

static int wtec_write_result_file(
    const char *path,
    const wtec_payload_t *payload,
    const double *thickness_g_mean,
    const double *thickness_g_std,
    const int *thickness_p,
    const int *thickness_slice_count,
    const int *thickness_superslice_dim,
    const double *thickness_sector_g,
    const double *length_g_mean,
    const double *length_g_std,
    const int *length_p,
    const int *length_slice_count,
    const int *length_superslice_dim,
    const double *length_sector_g,
    int mpi_size,
    int n_orb,
    int max_p_eff,
    int max_slice_count,
    int max_superslice_dim,
    int transport_task_count) {
  FILE *fh = fopen(path, "w");
  int i;
  int sector_count = (strcmp(payload->mode, "periodic_transverse") == 0) ? payload->n_layers_y : 1;
  int mid_disorder = wtec_mfp_disorder_index(payload);
  if (fh == NULL) {
    fprintf(stderr, "failed to open result file %s: %s\n", path, strerror(errno));
    return -1;
  }
  fprintf(fh, "{\n");
  fprintf(fh, "  \"transport_results_raw\": {\n");
  fprintf(fh, "    \"engine\": \"rgf\",\n");
  fprintf(fh, "    \"mode\": \"%s\",\n", payload->mode);
  fprintf(fh, "    \"periodic_axis\": \"%s\",\n", (payload->periodic_axis != '\0') ? (char[2]){payload->periodic_axis, '\0'} : "");
  fprintf(fh, "    \"lead_axis\": \"%c\",\n", payload->lead_axis);
  fprintf(fh, "    \"thickness_axis\": \"%c\",\n", payload->thickness_axis);
  fprintf(fh, "    \"n_layers_x\": %d,\n", payload->n_layers_x);
  fprintf(fh, "    \"n_layers_y\": %d,\n", payload->n_layers_y);
  fprintf(fh, "    \"mfp_n_layers_z\": %d,\n", payload->mfp_n_layers_z);
  fprintf(fh, "    \"energy\": %.16g,\n", payload->energy);
  fprintf(fh, "    \"eta\": %.16g,\n", payload->eta);
  fprintf(fh, "    \"ky_fractions\": [");
  for (i = 0; i < sector_count; ++i) {
    if (i > 0) {
      fputs(", ", fh);
    }
    if (strcmp(payload->mode, "periodic_transverse") == 0) {
      fprintf(fh, "%.16g", ((double)i) / ((double)payload->n_layers_y));
    } else {
      fprintf(fh, "%.16g", 0.0);
    }
  }
  fprintf(fh, "],\n");
  fprintf(fh, "    \"disorder_strengths\": "); wtec_write_double_array(fh, payload->disorder_strengths, payload->n_disorder); fprintf(fh, ",\n");
  fprintf(fh, "    \"n_ensemble\": %d,\n", payload->n_ensemble);
  fprintf(fh, "    \"thicknesses\": "); wtec_write_int_array(fh, payload->thicknesses, payload->n_thickness); fprintf(fh, ",\n");
  fprintf(fh, "    \"thickness_G\": ");
  if (payload->n_disorder <= 1) {
    wtec_write_double_array(fh, thickness_g_mean, payload->n_thickness);
  } else {
    wtec_write_double_matrix(fh, thickness_g_mean, payload->n_disorder, payload->n_thickness);
  }
  fprintf(fh, ",\n");
  fprintf(fh, "    \"thickness_G_std\": ");
  if (payload->n_disorder <= 1) {
    wtec_write_double_array(fh, thickness_g_std, payload->n_thickness);
  } else {
    wtec_write_double_matrix(fh, thickness_g_std, payload->n_disorder, payload->n_thickness);
  }
  fprintf(fh, ",\n");
  fprintf(fh, "    \"thickness_p_eff\": "); wtec_write_int_array(fh, thickness_p, payload->n_thickness); fprintf(fh, ",\n");
  fprintf(fh, "    \"thickness_slice_count\": "); wtec_write_int_array(fh, thickness_slice_count, payload->n_thickness); fprintf(fh, ",\n");
  fprintf(fh, "    \"thickness_superslice_dim\": "); wtec_write_int_array(fh, thickness_superslice_dim, payload->n_thickness); fprintf(fh, ",\n");
  if (payload->n_disorder <= 1) {
    fprintf(fh, "    \"thickness_sector_G\": "); wtec_write_double_matrix(fh, thickness_sector_g, payload->n_thickness, sector_count); fprintf(fh, ",\n");
  } else {
    fprintf(fh, "    \"thickness_sector_G_by_disorder\": "); wtec_write_double_tensor3(fh, thickness_sector_g, payload->n_disorder, payload->n_thickness, sector_count); fprintf(fh, ",\n");
  }
  fprintf(fh, "    \"mfp_lengths\": "); wtec_write_int_array(fh, payload->mfp_lengths, payload->n_mfp_lengths); fprintf(fh, ",\n");
  fprintf(fh, "    \"length_disorder_strength\": %.16g,\n", payload->disorder_strengths[mid_disorder]);
  fprintf(fh, "    \"length_G\": "); wtec_write_double_array(fh, length_g_mean, payload->n_mfp_lengths); fprintf(fh, ",\n");
  fprintf(fh, "    \"length_G_std\": "); wtec_write_double_array(fh, length_g_std, payload->n_mfp_lengths); fprintf(fh, ",\n");
  fprintf(fh, "    \"length_p_eff\": "); wtec_write_int_array(fh, length_p, payload->n_mfp_lengths); fprintf(fh, ",\n");
  fprintf(fh, "    \"length_slice_count\": "); wtec_write_int_array(fh, length_slice_count, payload->n_mfp_lengths); fprintf(fh, ",\n");
  fprintf(fh, "    \"length_superslice_dim\": "); wtec_write_int_array(fh, length_superslice_dim, payload->n_mfp_lengths); fprintf(fh, ",\n");
  fprintf(fh, "    \"length_sector_G\": "); wtec_write_double_matrix(fh, length_sector_g, payload->n_mfp_lengths, sector_count); fprintf(fh, "\n");
  fprintf(fh, "  },\n");
  fprintf(fh, "  \"runtime_cert\": {\n");
  fprintf(fh, "    \"engine\": \"rgf\",\n");
  fprintf(fh, "    \"binary_id\": \"%s\",\n", WTEC_RGF_BINARY_ID);
  fprintf(fh, "    \"numerical_status\": \"phase2_experimental\",\n");
  fprintf(fh, "    \"mode\": \"%s\",\n", payload->mode);
  fprintf(fh, "    \"periodic_axis\": \"%s\",\n", (payload->periodic_axis != '\0') ? (char[2]){payload->periodic_axis, '\0'} : "");
  fprintf(fh, "    \"queue\": \"%s\",\n", payload->queue);
  fprintf(fh, "    \"mpi_size\": %d,\n", mpi_size);
  fprintf(fh, "    \"omp_threads\": %d,\n", omp_get_max_threads());
  fprintf(fh, "    \"blas_backend\": \"%s\",\n", WTEC_RGF_BUILD_BLAS_BACKEND);
  fprintf(fh, "    \"n_orb\": %d,\n", n_orb);
  fprintf(fh, "    \"n_disorder\": %d,\n", payload->n_disorder);
  fprintf(fh, "    \"n_ensemble\": %d,\n", payload->n_ensemble);
  fprintf(fh, "    \"principal_layer_width\": %d,\n", max_p_eff);
  fprintf(fh, "    \"max_slice_count\": %d,\n", max_slice_count);
  fprintf(fh, "    \"max_superslice_dim\": %d,\n", max_superslice_dim);
  fprintf(fh, "    \"safe_rank_cap\": %d,\n", mpi_size);
  fprintf(
      fh,
      "    \"build_env\": {\"openmp_enabled\": %s, \"omp_max_threads\": %d, \"blas_backend\": \"%s\"},\n",
      wtec_json_bool(
#ifdef _OPENMP
          1
#else
          0
#endif
          ),
      omp_get_max_threads(),
      WTEC_RGF_BUILD_BLAS_BACKEND);
  fprintf(fh, "    \"transport_task_count\": %d\n", transport_task_count);
  fprintf(fh, "  }\n");
  fprintf(fh, "}\n");
  fclose(fh);
  return 0;
}

static void wtec_rgf_emit_probe_json(int rank, int size) {
  printf(
      "{\"binary_id\":\"%s\",\"probe_completed\":true,"
      "\"ready\":true,\"mpi_enabled\":true,\"rank\":%d,\"size\":%d,"
      "\"numerical_status\":\"phase2_experimental\","
      "\"blas_backend\":\"%s\","
      "\"build_env\":{\"openmp_enabled\":%s,\"omp_max_threads\":%d,\"blas_backend\":\"%s\"}}\n",
      WTEC_RGF_BINARY_ID,
      rank,
      size,
      WTEC_RGF_BUILD_BLAS_BACKEND,
      wtec_json_bool(
#ifdef _OPENMP
          1
#else
          0
#endif
          ),
      omp_get_max_threads(),
      WTEC_RGF_BUILD_BLAS_BACKEND);
  fflush(stdout);
}

int wtec_rgf_probe(wtec_rgf_probe_t *probe) {
  if (probe == NULL) {
    return 1;
  }
  probe->rank = 0;
  probe->size = 1;
  probe->mpi_enabled = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &probe->rank);
  MPI_Comm_size(MPI_COMM_WORLD, &probe->size);
  return 0;
}

static int wtec_build_transport_tasks(
    const wtec_payload_t *payload,
    wtec_transport_task_t **tasks_out,
    int *count_out) {
  int count = 0;
  int cursor = 0;
  int sector_count;
  int d, ens, i, sector;
  int mid_disorder;
  wtec_transport_task_t *tasks = NULL;
  if (payload == NULL || tasks_out == NULL || count_out == NULL) {
    return -1;
  }
  *tasks_out = NULL;
  *count_out = 0;
  sector_count = (strcmp(payload->mode, "periodic_transverse") == 0) ? payload->n_layers_y : 1;
  if (strcmp(payload->mode, "block_validation") == 0) {
    count = payload->n_thickness;
    if (count <= 0) {
      return 0;
    }
    tasks = (wtec_transport_task_t *)wtec_calloc((size_t)count, sizeof(wtec_transport_task_t));
    for (i = 0; i < payload->n_thickness; ++i) {
      tasks[i].point_kind = 0;
      tasks[i].point_index = i;
      tasks[i].point_total = payload->n_thickness;
      tasks[i].disorder_index = 0;
      tasks[i].ensemble_index = 0;
      tasks[i].sector_index = 0;
      tasks[i].sector_count = 1;
      tasks[i].nx = payload->n_layers_x;
      tasks[i].ny = payload->n_layers_y;
      tasks[i].nz = payload->thicknesses[i];
      tasks[i].seed = payload->base_seed;
      tasks[i].disorder_strength = 0.0;
    }
    *tasks_out = tasks;
    *count_out = count;
    return 0;
  }

  mid_disorder = wtec_mfp_disorder_index(payload);
  for (d = 0; d < payload->n_disorder; ++d) {
    const int eff = wtec_effective_ensemble_count(payload->disorder_strengths[d], payload->n_ensemble);
    count += sector_count * payload->n_thickness * eff;
  }
  if (payload->n_mfp_lengths > 0) {
    const int eff = wtec_effective_ensemble_count(payload->disorder_strengths[mid_disorder], payload->n_ensemble);
    count += sector_count * payload->n_mfp_lengths * eff;
  }
  if (count <= 0) {
    return 0;
  }
  tasks = (wtec_transport_task_t *)wtec_calloc((size_t)count, sizeof(wtec_transport_task_t));
  for (d = 0; d < payload->n_disorder; ++d) {
    const int eff = wtec_effective_ensemble_count(payload->disorder_strengths[d], payload->n_ensemble);
    for (ens = 0; ens < eff; ++ens) {
      const int seed = payload->base_seed + ens;
      for (i = 0; i < payload->n_thickness; ++i) {
        for (sector = 0; sector < sector_count; ++sector) {
          wtec_transport_task_t *task = &tasks[cursor++];
          task->point_kind = 0;
          task->point_index = i;
          task->point_total = payload->n_thickness;
          task->disorder_index = d;
          task->ensemble_index = ens;
          task->sector_index = sector;
          task->sector_count = sector_count;
          task->nx = payload->n_layers_x;
          task->ny = payload->n_layers_y;
          task->nz = payload->thicknesses[i];
          task->seed = seed;
          task->disorder_strength = payload->disorder_strengths[d];
        }
      }
    }
  }
  if (payload->n_mfp_lengths > 0) {
    const double disorder_strength = payload->disorder_strengths[mid_disorder];
    const int eff = wtec_effective_ensemble_count(disorder_strength, payload->n_ensemble);
    for (ens = 0; ens < eff; ++ens) {
      const int seed = payload->base_seed + ens;
      for (i = 0; i < payload->n_mfp_lengths; ++i) {
        for (sector = 0; sector < sector_count; ++sector) {
          wtec_transport_task_t *task = &tasks[cursor++];
          task->point_kind = 1;
          task->point_index = i;
          task->point_total = payload->n_mfp_lengths;
          task->disorder_index = mid_disorder;
          task->ensemble_index = ens;
          task->sector_index = sector;
          task->sector_count = sector_count;
          task->nx = payload->mfp_lengths[i];
          task->ny = payload->n_layers_y;
          task->nz = payload->mfp_n_layers_z;
          task->seed = seed;
          task->disorder_strength = disorder_strength;
        }
      }
    }
  }
  *tasks_out = tasks;
  *count_out = count;
  return 0;
}

int main(int argc, char **argv) {
  int rank = 0;
  int size = 1;
  int rc = MPI_Init(&argc, &argv);
  int errflag_local = 0;
  int errflag_global = 0;
  char errbuf[512];
  wtec_payload_t payload;
  wtec_hr_model_t model;
  wtec_progress_t progress;
  wtec_transport_task_t *tasks = NULL;
  double *thickness_samples_local = NULL, *thickness_samples_global = NULL;
  double *thickness_g_mean = NULL, *thickness_g_std = NULL;
  double *length_samples_local = NULL, *length_samples_global = NULL;
  double *length_g_mean = NULL, *length_g_std = NULL;
  double *thickness_sector_local = NULL, *thickness_sector_global = NULL;
  double *length_sector_local = NULL, *length_sector_global = NULL;
  int *tp_local = NULL, *tp_global = NULL;
  int *tsc_local = NULL, *tsc_global = NULL, *tsd_local = NULL, *tsd_global = NULL;
  int *lp_local = NULL, *lp_global = NULL;
  int *lsc_local = NULL, *lsc_global = NULL, *lsd_local = NULL, *lsd_global = NULL;
  int max_p_local = 0, max_p_global = 0;
  int max_slice_local = 0, max_slice_global = 0;
  int max_dim_local = 0, max_dim_global = 0;
  int sector_count = 1;
  int mid_disorder = 0;
  int thickness_sample_count = 0;
  int thickness_sector_count = 0;
  int length_sample_count = 0;
  int length_sector_count = 0;
  int transport_task_count = 0;
  int u;

  if (rc != MPI_SUCCESS) {
    fprintf(stderr, "MPI_Init failed with rc=%d\n", rc);
    return 1;
  }
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  (void)wtec_rgf_model_touch();
  (void)wtec_rgf_hamiltonian_touch();
  (void)wtec_rgf_sancho_touch();
  (void)wtec_rgf_sweep_touch();
  (void)wtec_rgf_periodic_touch();
  (void)wtec_rgf_disorder_touch();
  (void)wtec_rgf_payload_touch();
  (void)wtec_rgf_result_touch();
  (void)wtec_rgf_linalg_touch();
  (void)wtec_rgf_rng_touch();

  if (argc == 2 && strcmp(argv[1], "--probe") == 0) {
    if (rank == 0) {
      wtec_rgf_emit_probe_json(rank, size);
    }
    MPI_Finalize();
    return 0;
  }
  if (argc != 3) {
    if (rank == 0) {
      fprintf(stderr, "usage: wtec_rgf_runner --probe | <payload.json> <output.json>\n");
    }
    MPI_Finalize();
    return 2;
  }

  memset(&payload, 0, sizeof(payload));
  memset(&model, 0, sizeof(model));
  memset(&progress, 0, sizeof(progress));
  errbuf[0] = '\0';

  if (wtec_parse_payload_file(argv[1], &payload, errbuf, sizeof(errbuf)) != 0) {
    if (rank == 0) {
      fprintf(stderr, "payload parse failed: %s\n", errbuf);
    }
    MPI_Finalize();
    return 2;
  }
  if (payload.expected_mpi_np > 0 && payload.expected_mpi_np != size) {
    if (rank == 0) {
      fprintf(stderr, "expected MPI size %d, got %d\n", payload.expected_mpi_np, size);
    }
    wtec_free_payload(&payload);
    MPI_Finalize();
    return 2;
  }
  if (strcmp(payload.mode, "block_validation") != 0) {
    if (wtec_parse_hr_file(payload.hr_dat_path, &model, errbuf, sizeof(errbuf)) != 0) {
      if (rank == 0) {
        fprintf(stderr, "hr parse failed: %s\n", errbuf);
      }
      wtec_free_payload(&payload);
      MPI_Finalize();
      return 2;
    }
  }

  if (strcmp(payload.mode, "periodic_transverse") == 0) {
    for (u = 0; u < payload.n_thickness; ++u) {
      int p_eff = wtec_p_eff_for_model_nz(&model, payload.thicknesses[u]);
      if (payload.n_layers_x % p_eff != 0) {
        if (rank == 0) {
          fprintf(
              stderr,
              "phase1 RGF requires transport_n_layers_x=%d to be divisible by principal_layer_width=%d "
              "for thickness_uc=%d\n",
              payload.n_layers_x,
              p_eff,
              payload.thicknesses[u]);
        }
        wtec_free_hr_model(&model);
        wtec_free_payload(&payload);
        MPI_Finalize();
        return 2;
      }
    }
    if (payload.n_mfp_lengths > 0) {
      int p_eff_mfp = wtec_p_eff_for_model_nz(&model, payload.mfp_n_layers_z);
      for (u = 0; u < payload.n_mfp_lengths; ++u) {
        if (payload.mfp_lengths[u] % p_eff_mfp != 0) {
          if (rank == 0) {
            fprintf(
                stderr,
                "phase1 RGF requires mfp_length_uc=%d to be divisible by principal_layer_width=%d "
                "for mfp_n_layers_z=%d\n",
                payload.mfp_lengths[u],
                p_eff_mfp,
                payload.mfp_n_layers_z);
          }
          wtec_free_hr_model(&model);
          wtec_free_payload(&payload);
          MPI_Finalize();
          return 2;
        }
      }
    }
  }

  sector_count = (strcmp(payload.mode, "periodic_transverse") == 0) ? payload.n_layers_y : 1;
  mid_disorder = wtec_mfp_disorder_index(&payload);
  if (wtec_build_transport_tasks(&payload, &tasks, &transport_task_count) != 0 ||
      transport_task_count <= 0) {
    if (rank == 0) {
      fprintf(stderr, "failed to build transport task table\n");
    }
    wtec_free_hr_model(&model);
    wtec_free_payload(&payload);
    MPI_Finalize();
    return 2;
  }
  wtec_progress_init(&progress, &payload, rank, size);
  wtec_progress_emit_runner_event(&progress, "worker_start", &payload, transport_task_count);
  wtec_progress_emit_runner_event(&progress, "transport_run_start", &payload, transport_task_count);

  thickness_sample_count = payload.n_disorder * payload.n_thickness * payload.n_ensemble;
  thickness_sector_count = payload.n_disorder * payload.n_thickness * sector_count;
  length_sample_count = payload.n_mfp_lengths * payload.n_ensemble;
  length_sector_count = payload.n_mfp_lengths * sector_count;

  if (thickness_sample_count > 0) {
    thickness_samples_local = (double *)wtec_calloc((size_t)thickness_sample_count, sizeof(double));
    thickness_samples_global = (double *)wtec_calloc((size_t)thickness_sample_count, sizeof(double));
  }
  if (thickness_sector_count > 0) {
    thickness_sector_local = (double *)wtec_calloc((size_t)thickness_sector_count, sizeof(double));
    thickness_sector_global = (double *)wtec_calloc((size_t)thickness_sector_count, sizeof(double));
  }
  if (payload.n_thickness > 0) {
    tp_local = (int *)wtec_calloc((size_t)payload.n_thickness, sizeof(int));
    tp_global = (int *)wtec_calloc((size_t)payload.n_thickness, sizeof(int));
    tsc_local = (int *)wtec_calloc((size_t)payload.n_thickness, sizeof(int));
    tsc_global = (int *)wtec_calloc((size_t)payload.n_thickness, sizeof(int));
    tsd_local = (int *)wtec_calloc((size_t)payload.n_thickness, sizeof(int));
    tsd_global = (int *)wtec_calloc((size_t)payload.n_thickness, sizeof(int));
  }
  if (length_sample_count > 0) {
    length_samples_local = (double *)wtec_calloc((size_t)length_sample_count, sizeof(double));
    length_samples_global = (double *)wtec_calloc((size_t)length_sample_count, sizeof(double));
  }
  if (length_sector_count > 0) {
    length_sector_local = (double *)wtec_calloc((size_t)length_sector_count, sizeof(double));
    length_sector_global = (double *)wtec_calloc((size_t)length_sector_count, sizeof(double));
  }
  if (payload.n_mfp_lengths > 0) {
    lp_local = (int *)wtec_calloc((size_t)payload.n_mfp_lengths, sizeof(int));
    lp_global = (int *)wtec_calloc((size_t)payload.n_mfp_lengths, sizeof(int));
    lsc_local = (int *)wtec_calloc((size_t)payload.n_mfp_lengths, sizeof(int));
    lsc_global = (int *)wtec_calloc((size_t)payload.n_mfp_lengths, sizeof(int));
    lsd_local = (int *)wtec_calloc((size_t)payload.n_mfp_lengths, sizeof(int));
    lsd_global = (int *)wtec_calloc((size_t)payload.n_mfp_lengths, sizeof(int));
  }

  for (u = rank; u < transport_task_count; u += size) {
    const wtec_transport_task_t *task = tasks + u;
    const int point_kind = task->point_kind;
    const int local_index = task->point_index;
    const int sector_index = task->sector_index;
    const int nx = task->nx;
    const int ny = task->ny;
    const int nz = task->nz;
    const double ky_frac =
        ((double)sector_index) / ((double)((task->sector_count > 0) ? task->sector_count : 1));
    const double point_started_s = wtec_wall_seconds();
    wtec_point_result_t point;
    wtec_point_context_t point_ctx;

    point.p_eff = 0;
    point.slice_count = 0;
    point.superslice_dim = 0;
    point.transmission = 0.0;
    point_ctx.point_kind = point_kind;
    point_ctx.point_index = local_index;
    point_ctx.point_total = task->point_total;
    point_ctx.disorder_index = task->disorder_index;
    point_ctx.ensemble_index = task->ensemble_index;
    point_ctx.sector_index = sector_index;
    point_ctx.sector_count = task->sector_count;
    point_ctx.nx = nx;
    point_ctx.ny = ny;
    point_ctx.nz = nz;
    point_ctx.seed = task->seed;
    point_ctx.disorder_strength = task->disorder_strength;
    wtec_progress_emit_point_event(&progress, "native_point_start", &point_ctx, NULL, NULL, -1.0, -1, -1);

    if (strcmp(payload.mode, "periodic_transverse") == 0) {
      if (wtec_compute_transmission_periodic_y(
              &model,
              ky_frac,
              nx,
              nz,
              payload.energy,
              payload.eta,
              &progress,
              &point_ctx,
              &point) != 0) {
        errflag_local = 1;
        fprintf(
            stderr,
            "[rank %d] transport solve failed for kind=%d idx=%d disorder=%d ensemble=%d ky=%d nx=%d nz=%d\n",
            rank,
            point_kind,
            local_index,
            task->disorder_index,
            task->ensemble_index,
            sector_index,
            nx,
            nz);
        wtec_progress_emit_point_event(&progress, "worker_failed", &point_ctx, "periodic_y", NULL, -1.0, -1, -1);
        break;
      }
    } else if (strcmp(payload.mode, "block_validation") == 0) {
      if (wtec_compute_transmission_block_validation(&payload, &point) != 0) {
        errflag_local = 1;
        fprintf(stderr, "[rank %d] block-validation transport solve failed for idx=%d\n", rank, local_index);
        wtec_progress_emit_point_event(&progress, "worker_failed", &point_ctx, "block_validation", NULL, -1.0, -1, -1);
        break;
      }
    } else {
      if (wtec_compute_transmission_full_finite(
              &model,
              &payload,
              nx,
              ny,
              nz,
              payload.energy,
              payload.eta,
              &progress,
              &point_ctx,
              &point) != 0) {
        errflag_local = 1;
        fprintf(
            stderr,
            "[rank %d] full-finite transport solve failed for kind=%d idx=%d disorder=%d ensemble=%d nx=%d ny=%d nz=%d\n",
            rank,
            point_kind,
            local_index,
            task->disorder_index,
            task->ensemble_index,
            nx,
            ny,
            nz);
        wtec_progress_emit_point_event(&progress, "worker_failed", &point_ctx, "full_finite", NULL, -1.0, -1, -1);
        break;
      }
    }

    wtec_progress_emit_point_event(
        &progress,
        "native_point_done",
        &point_ctx,
        NULL,
        &point,
        wtec_wall_seconds() - point_started_s,
        -1,
        -1);

    if (point_kind == 0) {
      const size_t sample_idx =
          ((size_t)task->disorder_index * (size_t)payload.n_thickness + (size_t)local_index) *
              (size_t)payload.n_ensemble +
          (size_t)task->ensemble_index;
      const size_t sector_idx =
          ((size_t)task->disorder_index * (size_t)payload.n_thickness + (size_t)local_index) *
              (size_t)sector_count +
          (size_t)sector_index;
      if (thickness_samples_local != NULL) {
        thickness_samples_local[sample_idx] += point.transmission;
      }
      if (thickness_sector_local != NULL) {
        thickness_sector_local[sector_idx] += point.transmission;
      }
      if (tp_local != NULL && point.p_eff > tp_local[local_index]) tp_local[local_index] = point.p_eff;
      if (tsc_local != NULL && point.slice_count > tsc_local[local_index]) tsc_local[local_index] = point.slice_count;
      if (tsd_local != NULL && point.superslice_dim > tsd_local[local_index]) tsd_local[local_index] = point.superslice_dim;
    } else {
      const size_t sample_idx =
          (size_t)local_index * (size_t)payload.n_ensemble + (size_t)task->ensemble_index;
      const size_t sector_idx = (size_t)local_index * (size_t)sector_count + (size_t)sector_index;
      if (length_samples_local != NULL) {
        length_samples_local[sample_idx] += point.transmission;
      }
      if (length_sector_local != NULL) {
        length_sector_local[sector_idx] += point.transmission;
      }
      if (lp_local != NULL && point.p_eff > lp_local[local_index]) lp_local[local_index] = point.p_eff;
      if (lsc_local != NULL && point.slice_count > lsc_local[local_index]) lsc_local[local_index] = point.slice_count;
      if (lsd_local != NULL && point.superslice_dim > lsd_local[local_index]) lsd_local[local_index] = point.superslice_dim;
    }
    if (point.p_eff > max_p_local) max_p_local = point.p_eff;
    if (point.slice_count > max_slice_local) max_slice_local = point.slice_count;
    if (point.superslice_dim > max_dim_local) max_dim_local = point.superslice_dim;
  }

  MPI_Reduce(&errflag_local, &errflag_global, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
  if (errflag_global) {
    wtec_progress_emit_runner_event(&progress, "worker_failed", &payload, transport_task_count);
    wtec_progress_close(&progress);
    wtec_free_payload(&payload);
    wtec_free_hr_model(&model);
    free(tasks);
    free(thickness_samples_local); free(thickness_samples_global);
    free(length_samples_local); free(length_samples_global);
    free(thickness_sector_local); free(thickness_sector_global);
    free(length_sector_local); free(length_sector_global);
    free(tp_local); free(tp_global);
    free(tsc_local); free(tsc_global); free(tsd_local); free(tsd_global);
    free(lp_local); free(lp_global);
    free(lsc_local); free(lsc_global); free(lsd_local); free(lsd_global);
    MPI_Finalize();
    return 1;
  }

  if (thickness_sample_count > 0) {
    MPI_Reduce(thickness_samples_local, thickness_samples_global, thickness_sample_count, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  }
  if (length_sample_count > 0) {
    MPI_Reduce(length_samples_local, length_samples_global, length_sample_count, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  }
  if (thickness_sector_count > 0) {
    MPI_Reduce(thickness_sector_local, thickness_sector_global, thickness_sector_count, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  }
  if (length_sector_count > 0) {
    MPI_Reduce(length_sector_local, length_sector_global, length_sector_count, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  }
  if (payload.n_thickness > 0) {
    MPI_Reduce(tp_local, tp_global, payload.n_thickness, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(tsc_local, tsc_global, payload.n_thickness, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(tsd_local, tsd_global, payload.n_thickness, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
  }
  if (payload.n_mfp_lengths > 0) {
    MPI_Reduce(lp_local, lp_global, payload.n_mfp_lengths, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(lsc_local, lsc_global, payload.n_mfp_lengths, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(lsd_local, lsd_global, payload.n_mfp_lengths, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
  }
  MPI_Reduce(&max_p_local, &max_p_global, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&max_slice_local, &max_slice_global, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&max_dim_local, &max_dim_global, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    int d, i, ens, sector;
    if (payload.n_disorder * payload.n_thickness > 0) {
      thickness_g_mean =
          (double *)wtec_calloc((size_t)payload.n_disorder * (size_t)payload.n_thickness, sizeof(double));
      thickness_g_std =
          (double *)wtec_calloc((size_t)payload.n_disorder * (size_t)payload.n_thickness, sizeof(double));
    } else if (payload.n_disorder > 1) {
      thickness_g_mean = (double *)wtec_calloc(1u, sizeof(double));
      thickness_g_std = (double *)wtec_calloc(1u, sizeof(double));
    }
    if (payload.n_mfp_lengths > 0) {
      length_g_mean = (double *)wtec_calloc((size_t)payload.n_mfp_lengths, sizeof(double));
      length_g_std = (double *)wtec_calloc((size_t)payload.n_mfp_lengths, sizeof(double));
    }

    for (d = 0; d < payload.n_disorder; ++d) {
      const int eff = wtec_effective_ensemble_count(payload.disorder_strengths[d], payload.n_ensemble);
      for (i = 0; i < payload.n_thickness; ++i) {
        double sum = 0.0;
        double sumsq = 0.0;
        const size_t out_idx = (size_t)d * (size_t)payload.n_thickness + (size_t)i;
        for (ens = 0; ens < eff; ++ens) {
          const size_t sample_idx =
              ((size_t)d * (size_t)payload.n_thickness + (size_t)i) * (size_t)payload.n_ensemble +
              (size_t)ens;
          const double g = thickness_samples_global[sample_idx];
          sum += g;
          sumsq += g * g;
        }
        if (thickness_g_mean != NULL) {
          const double mean = sum / (double)eff;
          double var = sumsq / (double)eff - mean * mean;
          if (var < 0.0 && fabs(var) < 1.0e-12) {
            var = 0.0;
          }
          thickness_g_mean[out_idx] = mean;
          thickness_g_std[out_idx] = (eff > 1 && var > 0.0) ? sqrt(var) : 0.0;
        }
        if (thickness_sector_global != NULL) {
          for (sector = 0; sector < sector_count; ++sector) {
            const size_t sector_idx =
                ((size_t)d * (size_t)payload.n_thickness + (size_t)i) * (size_t)sector_count +
                (size_t)sector;
            thickness_sector_global[sector_idx] /= (double)eff;
          }
        }
      }
    }

    if (payload.n_mfp_lengths > 0) {
      const int eff = wtec_effective_ensemble_count(payload.disorder_strengths[mid_disorder], payload.n_ensemble);
      for (i = 0; i < payload.n_mfp_lengths; ++i) {
        double sum = 0.0;
        double sumsq = 0.0;
        for (ens = 0; ens < eff; ++ens) {
          const size_t sample_idx = (size_t)i * (size_t)payload.n_ensemble + (size_t)ens;
          const double g = length_samples_global[sample_idx];
          sum += g;
          sumsq += g * g;
        }
        if (length_g_mean != NULL) {
          const double mean = sum / (double)eff;
          double var = sumsq / (double)eff - mean * mean;
          if (var < 0.0 && fabs(var) < 1.0e-12) {
            var = 0.0;
          }
          length_g_mean[i] = mean;
          length_g_std[i] = (eff > 1 && var > 0.0) ? sqrt(var) : 0.0;
        }
        if (length_sector_global != NULL) {
          for (sector = 0; sector < sector_count; ++sector) {
            const size_t sector_idx = (size_t)i * (size_t)sector_count + (size_t)sector;
            length_sector_global[sector_idx] /= (double)eff;
          }
        }
      }
    }

    if (wtec_write_result_file(
            argv[2],
            &payload,
            thickness_g_mean,
            thickness_g_std,
            tp_global,
            tsc_global,
            tsd_global,
            thickness_sector_global,
            length_g_mean,
            length_g_std,
            lp_global,
            lsc_global,
            lsd_global,
            length_sector_global,
            size,
            (strcmp(payload.mode, "block_validation") == 0) ? 0 : model.num_wann,
            max_p_global,
            max_slice_global,
            max_dim_global,
            transport_task_count) != 0) {
      errflag_global = 1;
    }
    free(thickness_g_mean);
    free(thickness_g_std);
    free(length_g_mean);
    free(length_g_std);
  }
  MPI_Bcast(&errflag_global, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (errflag_global) {
    wtec_progress_emit_runner_event(&progress, "worker_failed", &payload, transport_task_count);
  } else {
    wtec_progress_emit_runner_event(&progress, "transport_run_done", &payload, transport_task_count);
    wtec_progress_emit_runner_event(&progress, "worker_done", &payload, transport_task_count);
  }

  wtec_free_payload(&payload);
  wtec_free_hr_model(&model);
  wtec_progress_close(&progress);
  free(tasks);
  free(thickness_samples_local); free(thickness_samples_global);
  free(length_samples_local); free(length_samples_global);
  free(thickness_sector_local); free(thickness_sector_global);
  free(length_sector_local); free(length_sector_global);
  free(tp_local); free(tp_global);
  free(tsc_local); free(tsc_global); free(tsd_local); free(tsd_global);
  free(lp_local); free(lp_global);
  free(lsc_local); free(lsc_global); free(lsd_local); free(lsd_global);

  MPI_Finalize();
  return errflag_global ? 1 : 0;
}
