#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_8bit_storage : require

#if USE_SUBGROUP_ADD || USE_SUBGROUP_ADD_NO_SHMEM
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#endif

#ifdef MUL_MAT_ID
#define EXPERT_COUNT 8
#endif

#include "mul_mat_vec_iface.glsl"

layout (push_constant) uniform parameter
{
    uint ncols;
    uint stride_a;
    uint stride_b;
    uint stride_d;

    uint batch_stride_a;
    uint batch_stride_b;
    uint batch_stride_d;

    uint fusion_flags;

#ifdef MUL_MAT_ID
    uint nei0;
    uint ne11;
    uint expert_i1;
    uint nbi1;
#else
    uint base_work_group_y;
    uint ne02;
    uint ne12;
    uint broadcast2;
    uint broadcast3;
    float norm_eps;     // NORM_FUSION only
    uint norm_partials; // NORM_FUSION only: number of RMS partials to read, 0 = compute the sum here
    uint norm_write;    // NORM_FUSION only: write the normalized row to binding 4
#endif
} p;

#ifdef MUL_MAT_ID
uint expert_id;
bool expert_skip; // an id of -1 skips the slot, the result row is zeroed
#endif

void get_offsets(out uint a_offset, out uint b_offset, out uint d_offset) {
#ifdef MUL_MAT_ID
    const uint expert_i0 = gl_WorkGroupID.y;
#else
    const uint batch_idx = gl_WorkGroupID.y + p.base_work_group_y;
#endif

#ifndef MUL_MAT_ID
    uint batch_idx_a = 0;
    if (batch_idx != 0) {
        const uint i13 = batch_idx / p.ne12;
        const uint i12 = batch_idx % p.ne12;

        const uint i03 = i13 / p.broadcast3;
        const uint i02 = i12 / p.broadcast2;

        batch_idx_a = i03 * p.ne02 + i02;
    }
#else
    const int expert = data_ids[expert_i0 + p.expert_i1 * p.nbi1];
    expert_skip = expert == -1;
    expert_id   = expert_skip ? 0 : uint(expert); // clamp, so that the offsets below stay in range
#endif

    a_offset =
#ifdef MUL_MAT_ID
            expert_id * (p.batch_stride_a / QUANT_K);
#else
            batch_idx_a * (p.batch_stride_a / QUANT_K);
#endif
    b_offset =
#ifdef MUL_MAT_ID
            (expert_i0 % p.ne11) * p.stride_b + p.expert_i1 * p.batch_stride_b;
#else
            batch_idx * p.batch_stride_b;
#endif
    d_offset =
#ifdef MUL_MAT_ID
            expert_i0 * p.stride_d + p.expert_i1 * p.batch_stride_d;
#else
            batch_idx * p.batch_stride_d;
#endif
}

layout (constant_id = 0) const uint BLOCK_SIZE = 32;
layout (constant_id = 1) const uint NUM_ROWS = 1;
layout (constant_id = 2) const uint NUM_COLS = 1;

#ifdef NORM_FUSION
// B is the raw RMS_NORM input: one contiguous row, ncols % 4 == 0. B loads apply (x * s) * w.
#if !USE_SUBGROUP_ADD_NO_SHMEM
shared float norm_sh[BLOCK_SIZE];
#endif
float norm_s;

float norm_workgroup_sum(float sum) {
#if USE_SUBGROUP_ADD_NO_SHMEM
    return subgroupAdd(sum);
#elif USE_SUBGROUP_ADD
    sum = subgroupAdd(sum);
    if (gl_SubgroupInvocationID == 0) {
        norm_sh[gl_SubgroupID] = sum;
    }
    barrier();
    sum = 0.0;
    for (uint s = 0; s < gl_NumSubgroups; ++s) {
        sum += norm_sh[s];
    }
    return sum;
#else
    const uint tid = gl_LocalInvocationID.x;
    norm_sh[tid] = sum;
    barrier();
    for (uint s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            norm_sh[tid] += norm_sh[tid + s];
        }
        barrier();
    }
    return norm_sh[0];
#endif
}

void norm_prologue() {
    const uint tid = gl_LocalInvocationID.x;
    const uint ncols4 = p.ncols / 4;
    float sum = 0.0;
    if (p.norm_partials > 0) {
        for (uint i = tid; i < p.norm_partials; i += BLOCK_SIZE) {
            sum += data_norm_partials[i];
        }
    } else {
        for (uint c = tid; c < ncols4; c += BLOCK_SIZE) {
            const vec4 x = vec4(data_b_v4[c]);
            sum += dot(x, x);
        }
    }
    sum = norm_workgroup_sum(sum);
    norm_s = inversesqrt(sum / float(p.ncols) + p.norm_eps);

    if (p.norm_write == 0) {
        return;
    }
    // each workgroup writes one slice of the normalized row
    const uint n_wg = gl_NumWorkGroups.x * gl_NumWorkGroups.z;
    const uint wg = gl_WorkGroupID.x + gl_NumWorkGroups.x * gl_WorkGroupID.z;
    const uint per_wg = (ncols4 + n_wg - 1) / n_wg;
    const uint c_end = min(ncols4, (wg + 1) * per_wg);
    for (uint c = wg * per_wg + tid; c < c_end; c += BLOCK_SIZE) {
        data_norm_out_v4[c] = (norm_s * vec4(data_b_v4[c])) * data_norm_w_v4[c];
    }
}

#define LOAD_B_V4(idx) ((norm_s * vec4(data_b_v4[idx])) * data_norm_w_v4[idx])
#define LOAD_B_V2(idx) ((norm_s * vec2(data_b_v2[idx])) * data_norm_w_v2[idx])
#else
#define LOAD_B_V4(idx) vec4(data_b_v4[idx])
#define LOAD_B_V2(idx) vec2(data_b_v2[idx])
#endif

#ifdef USE_SUBGROUP_ADD_NO_SHMEM
void reduce_result(inout FLOAT_TYPE temp[NUM_COLS][NUM_ROWS], const in uint32_t d_offset, const in uint32_t first_row, const in uint32_t num_rows, const in uint32_t tid) {
    [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
        [[unroll]] for (uint n = 0; n < num_rows; ++n) {
            temp[j][n] = subgroupAdd(temp[j][n]);
        }
    }

    if (tid == 0) {
        [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
            [[unroll]] for (uint n = 0; n < num_rows; ++n) {
#ifdef MUL_MAT_ID
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS0) != 0) {
                    temp[j][n] += FLOAT_TYPE(data_fuse0[expert_id*p.stride_d + first_row + n]);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_SCALE0) != 0) {
                    const uint expert_i0 = gl_GlobalInvocationID.y;
                    temp[j][n] *= FLOAT_TYPE(data_fuse0[expert_i0]);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_SCALE1) != 0) {
                    const uint expert_i0 = gl_GlobalInvocationID.y;
                    temp[j][n] *= FLOAT_TYPE(data_fuse1[expert_i0]);
                }
                if (expert_skip) {
                    temp[j][n] = FLOAT_TYPE(0);
                }
#else
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS0) != 0) {
                    temp[j][n] += FLOAT_TYPE(data_fuse0[j*p.batch_stride_d + d_offset + first_row + n]);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS1) != 0) {
                    temp[j][n] += FLOAT_TYPE(data_fuse1[j*p.batch_stride_d + d_offset + first_row + n]);
                }
#endif
                data_d[j*p.batch_stride_d + d_offset + first_row + n] = D_TYPE(temp[j][n]);
            }
        }
    }
}
#else
shared FLOAT_TYPE tmpsh[NUM_COLS][NUM_ROWS][BLOCK_SIZE];

void reduce_result(FLOAT_TYPE temp[NUM_COLS][NUM_ROWS], const in uint32_t d_offset, const in uint32_t first_row, const in uint32_t num_rows, const in uint32_t tid) {
    // subgroupAdd is probably faster on devices that support it,
    // particularly when the workgroup has more than one subgroup
#if USE_SUBGROUP_ADD
    // sum up partial sums within a subgroup
    [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
        [[unroll]] for (uint n = 0; n < num_rows; ++n) {
            temp[j][n] = subgroupAdd(temp[j][n]);
        }
    }

    // Go through shared memory to sum partials across subgroups
    if (gl_SubgroupInvocationID == 0) {
        [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
            [[unroll]] for (uint n = 0; n < num_rows; ++n) {
                tmpsh[j][n][gl_SubgroupID] = temp[j][n];
            }
        }
    }
    barrier();
    if (tid == 0) {
        [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
            [[unroll]] for (uint n = 0; n < num_rows; ++n) {
                temp[j][n] = FLOAT_TYPE(0);
                [[unroll]] for (uint s = 0; s < gl_NumSubgroups; ++s) {
                    temp[j][n] += tmpsh[j][n][s];
                }
#ifdef MUL_MAT_ID
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS0) != 0) {
                    temp[j][n] += FLOAT_TYPE(data_fuse0[expert_id*p.stride_d + first_row + n]);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_SCALE0) != 0) {
                    const uint expert_i0 = gl_GlobalInvocationID.y;
                    temp[j][n] *= FLOAT_TYPE(data_fuse0[expert_i0]);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_SCALE1) != 0) {
                    const uint expert_i0 = gl_GlobalInvocationID.y;
                    temp[j][n] *= FLOAT_TYPE(data_fuse1[expert_i0]);
                }
                if (expert_skip) {
                    temp[j][n] = FLOAT_TYPE(0);
                }
#else
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS0) != 0) {
                    temp[j][n] += FLOAT_TYPE(data_fuse0[j*p.batch_stride_d + d_offset + first_row + n]);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS1) != 0) {
                    temp[j][n] += FLOAT_TYPE(data_fuse1[j*p.batch_stride_d + d_offset + first_row + n]);
                }
#endif
                data_d[j*p.batch_stride_d + d_offset + first_row + n] = D_TYPE(temp[j][n]);
            }
        }
    }
#else
    // sum up partial sums and write back result
    [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
        [[unroll]] for (uint n = 0; n < num_rows; ++n) {
            tmpsh[j][n][tid] = temp[j][n];
        }
    }
    barrier();
    [[unroll]] for (uint s = BLOCK_SIZE/2; s > 0; s >>= 1) {
        if (tid < s) {
            [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
                [[unroll]] for (uint n = 0; n < num_rows; ++n) {
                    tmpsh[j][n][tid] += tmpsh[j][n][tid + s];
                }
            }
        }
        barrier();
    }
    if (tid == 0) {
        [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
            [[unroll]] for (uint n = 0; n < num_rows; ++n) {
#ifdef MUL_MAT_ID
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS0) != 0) {
                    tmpsh[j][n][0] += FLOAT_TYPE(data_fuse0[expert_id*p.stride_d + first_row + n]);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_SCALE0) != 0) {
                    const uint expert_i0 = gl_GlobalInvocationID.y;
                    tmpsh[j][n][0] *= FLOAT_TYPE(data_fuse0[expert_i0]);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_SCALE1) != 0) {
                    const uint expert_i0 = gl_GlobalInvocationID.y;
                    tmpsh[j][n][0] *= FLOAT_TYPE(data_fuse1[expert_i0]);
                }
                if (expert_skip) {
                    tmpsh[j][n][0] = FLOAT_TYPE(0);
                }
#else
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS0) != 0) {
                    tmpsh[j][n][0] += FLOAT_TYPE(data_fuse0[j*p.batch_stride_d + d_offset + first_row + n]);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS1) != 0) {
                    tmpsh[j][n][0] += FLOAT_TYPE(data_fuse1[j*p.batch_stride_d + d_offset + first_row + n]);
                }
#endif
                data_d[j*p.batch_stride_d + d_offset + first_row + n] = D_TYPE(tmpsh[j][n][0]);
            }
        }
    }
#endif
}
#endif
