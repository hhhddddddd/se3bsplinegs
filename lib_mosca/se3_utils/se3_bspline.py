import time
import torch
import numpy as np
from scipy.linalg import logm, expm


def R2q(R):
    """
    Convert a batch of rotation matrices to quaternions (w, x, y, z).
    Supports arbitrary batch dimensions, e.g., [N, K, 3, 3] -> [N, K, 4].
    
    :param R: torch.Tensor of shape [..., 3, 3]
    :return: torch.Tensor of shape [..., 4]
    """
    # 1. Record the original batch dimensions
    orig_shape = R.shape[:-2]
    # 2. Flatten the batch dimensions into a single one [B, 3, 3]
    R_flat = R.reshape(-1, 3, 3)
    B = R_flat.shape[0]
    
    # Calculate Trace
    tr = R_flat[:, 0, 0] + R_flat[:, 1, 1] + R_flat[:, 2, 2]
    
    # Initialize output quaternions [B, 4]
    q = torch.zeros((B, 4), device=R.device, dtype=R.dtype)
    
    # --- Case 1: Trace > 0 ---
    mask0 = tr > 0
    if mask0.any():
        s = torch.sqrt(tr[mask0] + 1.0 + 1e-10) * 2.0  # s = 4 * qw
        q[mask0, 0] = 0.25 * s
        q[mask0, 1] = (R_flat[mask0, 2, 1] - R_flat[mask0, 1, 2]) / s
        q[mask0, 2] = (R_flat[mask0, 0, 2] - R_flat[mask0, 2, 0]) / s
        q[mask0, 3] = (R_flat[mask0, 1, 0] - R_flat[mask0, 0, 1]) / s
        
    # --- Case 2: Trace <= 0 ---
    mask_else = ~mask0
    if mask_else.any():
        R_e = R_flat[mask_else]
        # Find the largest diagonal element to maintain numerical stability
        diag = torch.stack([R_e[:, 0, 0], R_e[:, 1, 1], R_e[:, 2, 2]], dim=1)
        max_i = torch.argmax(diag, dim=1)
        
        # Get global indices within the flattened array for the Trace <= 0 subset
        else_indices = mask_else.nonzero(as_tuple=True)[0]
        
        # Subcase: R[0,0] is the largest
        m0 = (max_i == 0)
        if m0.any():
            idx = else_indices[m0]
            R_sub = R_flat[idx]
            s = torch.sqrt(1.0 + R_sub[:, 0, 0] - R_sub[:, 1, 1] - R_sub[:, 2, 2]) * 2.0
            q[idx, 0] = (R_sub[:, 2, 1] - R_sub[:, 1, 2]) / s
            q[idx, 1] = 0.25 * s
            q[idx, 2] = (R_sub[:, 0, 1] + R_sub[:, 1, 0]) / s
            q[idx, 3] = (R_sub[:, 0, 2] + R_sub[:, 2, 0]) / s

        # Subcase: R[1,1] is the largest
        m1 = (max_i == 1)
        if m1.any():
            idx = else_indices[m1]
            R_sub = R_flat[idx]
            s = torch.sqrt(1.0 + R_sub[:, 1, 1] - R_sub[:, 0, 0] - R_sub[:, 2, 2]) * 2.0
            q[idx, 0] = (R_sub[:, 0, 2] - R_sub[:, 2, 0]) / s
            q[idx, 1] = (R_sub[:, 0, 1] + R_sub[:, 1, 0]) / s
            q[idx, 2] = 0.25 * s
            q[idx, 3] = (R_sub[:, 1, 2] + R_sub[:, 2, 1]) / s

        # Subcase: R[2,2] is the largest
        m2 = (max_i == 2)
        if m2.any():
            idx = else_indices[m2]
            R_sub = R_flat[idx]
            s = torch.sqrt(1.0 + R_sub[:, 2, 2] - R_sub[:, 0, 0] - R_sub[:, 1, 1]) * 2.0
            q[idx, 0] = (R_sub[:, 1, 0] - R_sub[:, 0, 1]) / s
            q[idx, 1] = (R_sub[:, 0, 2] + R_sub[:, 2, 0]) / s
            q[idx, 2] = (R_sub[:, 1, 2] + R_sub[:, 2, 1]) / s
            q[idx, 3] = 0.25 * s
            
    # 3. Reshape back to the original batch dimensions
    return q.reshape(*orig_shape, 4)

class SE3BSplineUltraFast:
    def __init__(self, track_3d, state):
        """
        Cumulative cubic SE(3) B-spline.
        Assumes each track has exactly the same number of valid control points.
        :param track_3d: [T, M, 4, 4]
        :param state: [T, M] (bool), sum(0) must be constant across all M columns.
        """
        self.device = track_3d.device
        self.dtype = track_3d.dtype
        self.T, self.M = state.shape
        self.all_timestamps = torch.arange(self.T, device=self.device, dtype=self.dtype)

        # NOTE Each track MUST have the same number of True values.
        self.V = int(state[:, 0].sum())

        _, valid_time_indices = torch.where(state.transpose(0, 1))
        self.padded_indices = valid_time_indices.view(self.M, self.V).to(self.dtype)

        track_3d_permuted = track_3d.permute(1, 0, 2, 3) # M,T,4,4  
        self.padded_poses = track_3d_permuted[state.transpose(0, 1)].view(self.M, self.V, 4, 4) # M,V,4,4

        if self.V > 1:
            poses_start = self.padded_poses[:, :-1]
            poses_end = self.padded_poses[:, 1:]
            self.padded_xis = self._se3_log_from_pose_pairs(poses_start, poses_end)

            # Extrapolate one virtual control pose before the first pose so the
            # boundary interval can still use the cubic cumulative formulation.
            self.first_segment_base = self.padded_poses[:, 0] @ torch.linalg.matrix_exp(-self.padded_xis[:, 0])
        else:
            self.padded_xis = torch.zeros((self.M, 0, 4, 4), device=self.device, dtype=self.dtype)
            self.first_segment_base = self.padded_poses[:, 0]

    def _cumulative_cubic_basis(self, u):
        """
        Hybrid cubic cumulative basis SE(3) b-spline (first-order and third-order).
        u = 1 if t_valid is not adjacent to t_obv and t_valid < t_obv
        u = omega_1 if t_valid and t_obv are separated by a valid timestamp and t_valid < t_obv NOTE base pose
        u = omega_2 if t_valid is adjacent to t_obv and t_valid < t_obv
        u = omega_3 if t_valid is adjacent to t_obv and t_valid > t_obv
        u = 0 if t_valid > t_obv

        pose continuity, velocity continuity, acceleration continuity
        u: [B, 1, 1] in [0, 1]
        """
        u2 = u * u
        u3 = u2 * u
        omega_1 = (u3 - 3.0 * u2 + 3.0 * u + 5.0) / 6.0 
        omega_2 = (-2.0 * u3 + 3.0 * u2 + 3.0 * u + 1.0) / 6.0 
        omega_3 = u3 / 6.0 
        return omega_1, omega_2, omega_3

    def _se3_log_batch(self, T_batch):
        """Vectorized SE(3) log map."""
        N = T_batch.shape[0]
        device = T_batch.device
        dtype = T_batch.dtype

        R = T_batch[:, :3, :3]
        t = T_batch[:, :3, 3].unsqueeze(-1)

        cos_theta = (torch.diagonal(R, dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0
        eps = 1e-7

        ln_T = torch.zeros((N, 4, 4), device=device, dtype=dtype)
        omega_skew = torch.zeros((N, 3, 3), device=device, dtype=dtype)
        eye3 = torch.eye(3, device=device, dtype=dtype)
        V_inv = eye3.unsqueeze(0).repeat(N, 1, 1)

        # NOTE Avoid backprop through acos near theta ~= 0 to prevent NaN gradients
        small_mask = cos_theta > (1.0 - 1e-6)
        large_mask = ~small_mask

        if small_mask.any():
            R_s = R[small_mask]
            omega_s = (R_s - R_s.transpose(-1, -2)) / 2.0
            omega_skew[small_mask] = omega_s
            V_inv[small_mask] = eye3.unsqueeze(0) - 0.5 * omega_s + (1.0 / 12.0) * (omega_s @ omega_s)

        if large_mask.any():
            cos_theta_l = torch.clamp(cos_theta[large_mask], -1.0 + eps, 1.0 - eps)
            theta_l = torch.acos(cos_theta_l).view(-1, 1, 1)
            R_l = R[large_mask]
            sin_l = torch.clamp(torch.sin(theta_l), min=1e-6)
            omega_l = (theta_l / (2.0 * sin_l)) * (R_l - R_l.transpose(-1, -2))
            omega_skew[large_mask] = omega_l

            theta_sq = theta_l * theta_l
            a = sin_l / theta_l
            b = torch.clamp((1.0 - torch.cos(theta_l)) / theta_sq, min=1e-8)
            c = (1.0 - (a / (2.0 * b))) / theta_sq
            V_inv[large_mask] = eye3.unsqueeze(0) - 0.5 * omega_l + c * (omega_l @ omega_l)

        ln_T[:, :3, :3] = omega_skew
        ln_T[:, :3, 3] = (V_inv @ t).squeeze(-1)
        return ln_T

    def _se3_log_from_pose_pairs(self, poses_start, poses_end):
        """Compute batched log(inv(poses_start) @ poses_end) with arbitrary batch dims."""
        delta_Qs = torch.linalg.solve(poses_start, poses_end)
        flat_delta_Qs = delta_Qs.reshape(-1, 4, 4)
        return self._se3_log_batch(flat_delta_Qs).reshape_as(delta_Qs)

    def _pruned_candidate_cost(self, candidate_idx, query_t, track_chunk_size):
        total_cost = torch.zeros((), device=self.device, dtype=self.dtype)

        for start in range(0, self.M, track_chunk_size):
            end = min(start + track_chunk_size, self.M)

            p_prev = self.padded_poses[start:end, candidate_idx - 1]
            p_next = self.padded_poses[start:end, candidate_idx + 1]
            xi_curr = self._se3_log_from_pose_pairs(p_prev, p_next)

            if candidate_idx == 1:
                base_pose = self.padded_poses[start:end, 0] @ torch.linalg.matrix_exp(-xi_curr)
                xi_prev = xi_curr
            else:
                base_pose = self.padded_poses[start:end, candidate_idx - 2]
                xi_prev = self.padded_xis[start:end, candidate_idx - 2]

            if candidate_idx < self.V - 2:
                xi_next = self.padded_xis[start:end, candidate_idx + 1]
            else:
                xi_next = xi_curr

            t_start = self.padded_indices[start:end, candidate_idx - 1]
            t_end = self.padded_indices[start:end, candidate_idx + 1]
            dt = (t_end - t_start).clamp(min=1e-6)
            u = ((query_t - t_start) / dt).reshape(-1, 1, 1).clamp(0.0, 1.0)

            omega_1, omega_2, omega_3 = self._cumulative_cubic_basis(u)
            p_pruned = (
                base_pose
                @ torch.linalg.matrix_exp(omega_1 * xi_prev)
                @ torch.linalg.matrix_exp(omega_2 * xi_curr)
                @ torch.linalg.matrix_exp(omega_3 * xi_next)
            )

            p_curr = self.padded_poses[start:end, candidate_idx]
            dist_t = torch.sum((p_curr[:, :3, 3] - p_pruned[:, :3, 3]) ** 2, dim=-1)

            R_diff = p_curr[:, :3, :3].transpose(-1, -2) @ p_pruned[:, :3, :3]
            trace_diff = torch.diagonal(R_diff, dim1=-1, dim2=-2).sum(-1)
            dist_r = (3.0 - trace_diff) / 2.0

            total_cost = total_cost + (dist_t + dist_r).sum()

        return total_cost

    def _evaluate_cubic_segments(self, track_ids, segment_idx, u):
        """
        Evaluate cumulative SE(3) b-spline.

        cumulative SE(3) b-spline:
            T(t) = exp(u_0*xi_0) exp(u_1*xi_1) ... exp(u_{N_c-1}*xi_{N_c-1})
        based on the Hybrid cubic cumulative basis functions, we can simplify it to:
            T(t) = Q_base exp(u_prev*xi_prev) exp(u_curr*xi_curr) exp(u_next*xi_next)

        The boundary segments are completed by repeating the first or last
        relative increment and extrapolating one virtual base pose before the
        first valid control pose.
        """
        if self.V == 1:
            return self.padded_poses[track_ids, 0]

        base_idx = torch.clamp(segment_idx - 1, min=0)
        base_pose = self.padded_poses[track_ids, base_idx].clone() # prev pose
        first_mask = segment_idx == 0
        if first_mask.any(): # process base pose specically
            base_pose[first_mask] = self.first_segment_base[track_ids[first_mask]]

        xi_prev_idx = torch.clamp(segment_idx - 1, min=0)
        xi_next_idx = torch.clamp(segment_idx + 1, max=self.V - 2)
        xi_prev = self.padded_xis[track_ids, xi_prev_idx] # prev xi
        xi_curr = self.padded_xis[track_ids, segment_idx] # curr xi
        xi_next = self.padded_xis[track_ids, xi_next_idx] # next xi

        omega_1, omega_2, omega_3 = self._cumulative_cubic_basis(u) # Hybrid basis functions representation
        exp_1 = torch.linalg.matrix_exp(omega_1 * xi_prev)
        exp_2 = torch.linalg.matrix_exp(omega_2 * xi_curr)
        exp_3 = torch.linalg.matrix_exp(omega_3 * xi_next)
        return base_pose @ exp_1 @ exp_2 @ exp_3

    def query_poses(self, query_times, query_track_num):
        """
        Query cumulative cubic SE(3) B-spline poses.
        """
        N, K = query_track_num.shape
        flat_times = query_times.to(self.padded_indices.dtype).reshape(N, 1).expand(N, K).reshape(-1, 1) # N*K,1 range 0 ~ T
        flat_track_num = query_track_num.reshape(-1) # N*K range  0 ~ M

        if self.V == 1:
            out_poses = self.padded_poses[flat_track_num, 0].reshape(N, K, 4, 4)
            t_out = out_poses[..., :3, 3]
            quat_out = R2q(out_poses[..., :3, :3])
            return t_out, quat_out

        B_size = flat_times.shape[0]
        m_indices = self.padded_indices[flat_track_num] # N*K, V
        idx = torch.searchsorted(m_indices, flat_times).squeeze(-1) - 1 # N*K, NOTE first/start control point index
        idx = idx.clamp(min=0, max=self.V - 2)

        batch_range = torch.arange(B_size, device=self.device)
        t_start = m_indices[batch_range, idx] # N*K,
        t_end = m_indices[batch_range, idx + 1] # N*K,
        dt = (t_end - t_start).clamp(min=1e-6)
        u = ((flat_times.squeeze(-1) - t_start) / dt).reshape(-1, 1, 1).clamp(0.0, 1.0)

        interp = self._evaluate_cubic_segments(flat_track_num, idx, u)

        out_poses = interp.reshape(N, K, 4, 4)
        t_out = out_poses[..., :3, 3]
        quat_out = R2q(out_poses[..., :3, :3])
        return t_out, quat_out

    @torch.no_grad()
    def prune_control_points(self, track_chunk_size=4096):
        """
        Find the best timestamp 't' to prune that minimizes the trajectory change.
        :return: The global timestamp index t to be pruned.
        """
        # 1. Get current active timestamps
        active_indices = self.padded_indices[0].long() # V
        V = self.V
        
        if V <= 2:
            return None # Cannot prune if only start and end points remain

        costs = torch.empty(V - 2, device=self.device, dtype=self.dtype)
        
        # 2. Evaluate each candidate without rebuilding the full pruned spline.
        for cost_idx, t_curr_idx in enumerate(range(1, V - 1)):
            query_t = self.padded_indices[0, t_curr_idx].to(self.dtype)
            costs[cost_idx] = self._pruned_candidate_cost(
                candidate_idx=t_curr_idx,
                query_t=query_t,
                track_chunk_size=track_chunk_size,
            )

        # 5. Find the candidate with the minimum total cost
        best_candidate_idx = torch.argmin(costs)

        if costs[best_candidate_idx] > 5.0: # 1e-2
            return None
        
        # 6. Return the global timestamp index corresponding to the best candidate
        best_t = active_indices[best_candidate_idx + 1]
        
        return best_t.item()
