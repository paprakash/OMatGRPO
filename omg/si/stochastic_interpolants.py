from typing import Callable, List, Tuple, Union, Sequence, Dict
from tqdm import trange
import torch
from torch_geometric.data import Data
from omg.globals import SMALL_TIME, BIG_TIME, MAX_ATOM_NUM
from omg.utils import reshape_t, DataField
from .abstracts import StochasticInterpolant
from .single_stochastic_interpolant import DifferentialEquationType


class StochasticInterpolants(object):
    """
    Collection of several stochastic interpolants between points x_0 and x_1 from two distributions p_0 and
    p_1 at times t for different coordinate types x (like atom species, fractional coordinates, and lattice vectors).

    Every stochastic interpolant is associated with a data field and a cost factor. The possible data fields are defined
    in the omg.utils.DataField enumeration. Data is transmitted using the torch_geometric.data.Data class which allows
    for accessing the data with a dictionary-like interface.

    The loss returned by every stochastic interpolant is scaled by the corresponding cost factor.

    :param stochastic_interpolants:
        Sequence of stochastic interpolants for the different coordinate types.
    :type stochastic_interpolants: Sequence[StochasticInterpolant]
    :param data_fields:
        Sequence of data fields for the different stochastic interpolants.
    :type data_fields: Sequence[str]
    :param integration_time_steps:
        Number of integration time steps for the integration of the collection of stochastic interpolants.

    :raises ValueError:
        If the number of stochastic interpolants and costs are not equal.
        If the number of stochastic interpolants and data fields are not equal.
        If the number of integration time steps is not positive.
    """

    def __init__(self, stochastic_interpolants: Sequence[StochasticInterpolant], data_fields: Sequence[str],
                 integration_time_steps: int, enable_progress_bar: bool = True) -> None:
        """Constructor of the StochasticInterpolants class."""
        super().__init__()
        if not len(stochastic_interpolants) == len(data_fields):
            raise ValueError("The number of stochastic interpolants and data fields must be equal.")
        try:
            self._data_fields = [DataField[data_field.lower()] for data_field in data_fields]
        except AttributeError:
            raise ValueError(f"All data fields must be in {[d.name for d in DataField]}.")

        if not integration_time_steps > 0:
            raise ValueError("The number of integration time steps must be positive.")
        self._stochastic_interpolants = stochastic_interpolants
        self._integration_time_steps = integration_time_steps
        self._enable_progress_bar = enable_progress_bar

    def __len__(self) -> int:
        """
        Return the number of stochastic interpolants handled by this class.

        :return:
            Number of stochastic interpolants.
        :rtype: int
        """
        return len(self._stochastic_interpolants)

    def loss_keys(self) -> List[str]:
        """
        Return the keys of the losses returned by this class.

        The keys of the losses are constructed by concatenating the data field name and the loss key of the
        corresponding stochastic interpolant.

        :return:
            Keys of the losses.
        :rtype: List[str]
        """
        loss_keys = []
        for df, si in zip(self._data_fields, self._stochastic_interpolants):
            for key in si.loss_keys():
                full_key = f"{df.name}_{key}"
                if full_key in loss_keys:
                    raise ValueError(f"Key {full_key} is already used as a loss key.")
                loss_keys.append(full_key)
        return loss_keys

    def _interpolate(self, t: torch.Tensor, x_0: Data, x_1: Data) -> tuple[Data, Data]:
        """
        Stochastically interpolate between the collection of points x_0 and x_1 from the collection of two distributions
        p_0 and p_1 at times t.

        :param t:
            Times in [0,1].
        :type t: torch.Tensor
        :param x_0:
            Collection of points from the collection of distributions p_0 stored in a torch_geometric.data.Data object.
        :type x_0: torch_geometric.data.Data
        :param x_1:
            Collection of points from the collection of distributions p_1 stored in a torch_geometric.data.Data object.
        :type x_1: torch_geometric.data.Data

        :return:
            Collection of stochastically interpolated points x_t stored in a torch_geometric.data.Data object,
            and the collection of z values stored in a torch_geometric.data.Data object.
        :rtype: tuple[torch_geometric.data.Data, torch_geometric.data.Data]
        """
        x_0_dict = x_0.to_dict()
        x_1_dict = x_1.to_dict()
        assert torch.equal(x_0.batch, x_1.batch)
        assert torch.equal(x_0.n_atoms, x_1.n_atoms)
        n_atoms = x_0.n_atoms
        x_t = x_0.clone()
        x_t_dict = x_t.to_dict()
        z_data = {}
        for stochastic_interpolant, data_field in zip(self._stochastic_interpolants, self._data_fields):
            assert data_field.name in x_0_dict
            assert data_field.name in x_1_dict
            assert data_field.name in x_t_dict
            reshaped_t = reshape_t(t, n_atoms, data_field)
            assert reshaped_t.shape == x_0_dict[data_field.name].shape
            # Cell data requires different batch indices.
            interpolated_x_t, z = stochastic_interpolant.interpolate(
                reshaped_t, x_0_dict[data_field.name], x_1_dict[data_field.name],
                x_0.batch if data_field != DataField.cell else torch.arange(len(x_0.n_atoms)))
            # Assignment does not update x_t.
            x_t_dict[data_field.name].copy_(interpolated_x_t)
            assert data_field.name not in z_data
            z_data[data_field.name] = z
        return x_t, Data.from_dict(z_data)

    def losses(self, model_function: Callable[[Data, torch.tensor], Data], t: torch.Tensor, x_0: Data,
               x_1: Data) -> dict[str, torch.Tensor]:
        """
        Compute the losses for the collection of stochastic interpolants between the collection of points x_0 and x_1
        from a collection of distributions p_0 and p_1 at times t based on the collection of model predictions for the
        velocity fields b and the denoisers eta.

        This function expects that the velocity b and denoiser eta corresponding to the data field data_field are stored
        with the keys data_field_b and data_field_eta in the model prediction.

        The losses are returned as a dictionary with the data field names as keys and the corresponding losses as
        values.

        :param model_function:
            Model function returning the collection of velocity fields b and the denoisers eta stored in a
            torch_geometric.data.Data object given the current collection of points x_t stored in a
            torch_geometric.data.Data object and times t.
        :type model_function: Callable[[torch_geometric.data.Data, torch.Tensor], torch_geometric.data.Data]
        :param t:
            Times in [0,1].
        :type t: torch.Tensor
        :param x_0:
            Collection of points from the collection of distributions p_0 stored in a torch_geometric.data.Data object.
        :type x_0: torch_geometric.data.Data
        :param x_1:
            Collection of points from the collection of distributions p_1 stored in a torch_geometric.data.Data object.
        :type x_1: torch_geometric.data.Data

        :return:
            The losses for the collection of stochastic interpolants.
        :rtype: dict[str, torch.Tensor]
        """
        # Interpolate everything first so that we can pass all interpolated to the model function.
        x_t, z = self._interpolate(t, x_0, x_1)

        x_0_dict = x_0.to_dict()
        x_1_dict = x_1.to_dict()
        x_t_dict = x_t.to_dict()
        z_dict = z.to_dict()
        assert torch.equal(x_0.batch, x_1.batch)
        assert torch.equal(x_0.n_atoms, x_1.n_atoms)
        n_atoms = x_0.n_atoms
        losses = {}
        for stochastic_interpolant, data_field in zip(self._stochastic_interpolants, self._data_fields):
            b_data_field = data_field.name + "_b"
            eta_data_field = data_field.name + "_eta"
            assert data_field.name in x_0_dict
            assert data_field.name in x_1_dict
            assert data_field.name in x_t_dict
            reshaped_t = reshape_t(t, n_atoms, data_field)
            assert reshaped_t.shape == x_0_dict[data_field.name].shape
            assert reshaped_t.shape == x_1_dict[data_field.name].shape
            assert reshaped_t.shape == x_t_dict[data_field.name].shape

            def model_prediction_fn(x):
                # Clone x_t inside the function so that this function can be called several time.
                # If cloned outside, torch will complain that one of the variables needed for gradient computation has
                # been modified by an inplace operation.
                x_t_clone = x_t.clone()
                x_t_clone_dict = x_t_clone.to_dict()
                x_t_clone_dict[data_field.name].copy_(x)
                # TODO: Cache return of model function.
                model_result = model_function(x_t_clone, t)
                return model_result[b_data_field], model_result[eta_data_field]

            assert data_field.name in z_dict
            assert "loss_" + data_field.name not in losses
            # Cell data requires different batch indices.
            l = stochastic_interpolant.loss(
                model_prediction_fn, reshaped_t, x_0_dict[data_field.name], x_1_dict[data_field.name],
                x_t_dict[data_field.name], z[data_field.name],
                x_0.batch if data_field != DataField.cell else torch.arange(len(x_0.n_atoms)))
            for l_key, l_value in l.items():
                assert l_key not in losses
                losses[f"{data_field.name}_{l_key}"] = l_value
        return losses

    def integrate(self, x_0: Data, model_function: Callable[[Data, torch.Tensor], Data],
                  save_intermediate: bool = False) -> Union[Data, Tuple[Data, List[Data]]]:
        """
        Integrate the collection of points x_0 from the collection of distributions p_0 from time 0 to 1 based on the
        model that provides the collection of velocity fields b and denoisers eta.

        In principle, every stochastic interpolant could be integrated independently. However, the model function
        expects the updated positions of all stochastic interpolants at the same time. In this version, the integration
        is discretized in time. Every stochastic interpolant is integrated independently until the next time step based
        on the collection of points x_0 at the last timestep.

        :param x_0:
            Collection of points from the collection of distributions p_0 stored in a torch_geometric.data.Data object.
        :type x_0: torch_geometric.data.Data
        :param model_function:
            Model function returning the collection of velocity fields b and the denoisers eta stored in a
            torch_geometric.data.Data object given the current collection of points x_t stored in a
            torch_geometric.data.Data object and times t.
        :type model_function: Callable[[torch_geometric.data.Data, torch.Tensor], torch_geometric.data.Data]
        :save_intermediate:
            If True, the intermediate points of the integration are saved and returned.
        :type save_intermediate: bool

        :return:
            Collection of integrated points x_1 stored in a torch_geometric.data.Data object.
            If save_intermediate is True, furthermore a list of the intermediate points in Data objects is returned.
        :rtype: torch_geometric.data.Data
        """
        times = torch.linspace(SMALL_TIME, BIG_TIME, self._integration_time_steps, device=x_0.pos.device)
        x_t = x_0.clone(*[data_field.name for data_field in self._data_fields])
        new_x_t = x_0.clone(*[data_field.name for data_field in self._data_fields])
        x_t_dict = x_t.to_dict()
        new_x_t_dict = new_x_t.to_dict()
        assert all(data_field.name in x_t_dict for data_field in self._data_fields)
        assert all(data_field.name in new_x_t_dict for data_field in self._data_fields)

        if save_intermediate:
            inter_list = [x_t]
        else:
            inter_list = None
        for t_index in trange(1, len(times), desc='Integrating', disable=not self._enable_progress_bar):
            t = times[t_index - 1]
            dt = times[t_index] - times[t_index - 1]
            for stochastic_interpolant, data_field in zip(self._stochastic_interpolants, self._data_fields):
                b_data_field = data_field.name + "_b"
                eta_data_field = data_field.name + "_eta"
                x_int = x_t.clone(*[data_field.name for data_field in self._data_fields])
                x_int_dict = x_int.to_dict()

                def model_prediction_fn(time, x):
                    # The model expects the time to be repeated for every element in the batch.
                    # The time argument, however, is just a zero-dimensional tensor.
                    time = time.repeat(len(x_int_dict['n_atoms']))
                    x_int_dict[data_field.name].copy_(x)
                    model_result = model_function(x_int, time)
                    return model_result[b_data_field], model_result[eta_data_field]

                # Do not use x_int_dict[data_field.name] here because it will be implicitly updated in the
                # model_prediction_fn, which leads to unpredictable bugs.
                # Cell data requires different batch indices.
                new_x_t_dict[data_field.name].copy_(stochastic_interpolant.integrate(
                    model_prediction_fn, x_t_dict[data_field.name], t, dt,
                    x_0.batch if data_field != DataField.cell else torch.arange(len(x_0.n_atoms))))

            x_t = new_x_t.clone(*[data_field.name for data_field in self._data_fields])
            x_t_dict = x_t.to_dict()
            if save_intermediate:
                inter_list.append(x_t)
        if save_intermediate:
            return x_t, inter_list
        else:
            return x_t
        
    @torch.no_grad()
    def integrate_with_logprob(
        self,
        x_0: Data,
        model: "Model",
        model_ref: "Model" = None,
        fields: Tuple[str, ...] = ("pos", "cell"),
        stochastic: bool = True,
        return_trajectory: bool = True,
    ) -> Tuple[Data, Dict[str, torch.Tensor], torch.Tensor]:
        """
        RL-only integrator: explicit Euler–Maruyama SDE stepping for specified fields (pos/cell),
        with per-step Gaussian log-probabilities and a compact replay trajectory.

        Returns:
          gen            OMGData at final time (same batch B),
          traj           dict containing times, dt, initial states and per-step next states, and static batch info,
          logp_dict      dict of per-field log-probabilities:
                             "pos":     FloatTensor [B, T]   (if "pos" in fields)
                             "cell":    FloatTensor [B, T]   (if "cell" in fields)
                             "species": {"logp": [E], "step_idx": [E], "struct_idx": [E]}
                                        (if the species start all-masked).
        """
        # Time grid
        device = x_0.pos.device
        times = torch.linspace(SMALL_TIME, BIG_TIME, self._integration_time_steps, device=device)
        T = times.numel() - 1  # number of steps

        # Build fast lookup for field -> stochastic interpolant
        # (We access epsilon/gamma/corrector and velocity annealing from each field's SI.)
        field2si = {df.name: si for df, si in zip(self._data_fields, self._stochastic_interpolants)}
        # Ensure requested fields use SDE or DISCRETE (DFM species); epsilon/gamma only
        # required for SDE. DISCRETE (DFM) uses _noise and _mask_index instead.
        for fname in fields:
            si = field2si.get(fname, None)
            assert si is not None, f"Requested field '{fname}' not found among stochastic interpolants."
            de_type = getattr(si, "_differential_equation_type", None)
            assert de_type in (DifferentialEquationType.SDE, DifferentialEquationType.DISCRETE), \
                f"Field '{fname}' must use SDE or DISCRETE for RL (got {de_type})."
            if de_type == DifferentialEquationType.SDE:
                assert getattr(si, "_epsilon", None) is not None and getattr(si, "_gamma", None) is not None, \
                    f"SDE field '{fname}' must define epsilon and gamma for RL."

        # Species evolution gate: decide ONCE at t=0 whether species should evolve each step.
        # All-mask start -> evolve species every step, so the model sees a joint state like the
        # ones it was trained on. Resolved species (fixed composition) -> species stay pinned.
        si_species = field2si.get("species", None)
        species_starts_masked = False
        species_source_model = None
        if si_species is not None:
            species_raw = x_0.species
            any_masked = bool((species_raw == 0).any().item())
            all_masked = bool((species_raw == 0).all().item())
            if any_masked and not all_masked:
                raise ValueError(
                    "integrate_with_logprob: x_0.species has mixed mask states "
                    "(some Z=0, some non-zero). Must be either all-masked (species generated) "
                    "or fully resolved (fixed composition)."
                )
            species_starts_masked = all_masked
            if species_starts_masked:
                if "species" in fields:
                    species_source_model = model
                else:
                    assert model_ref is not None, (
                        "An all-masked species start with species not in fields "
                        "needs model_ref to generate the species."
                    )
                    species_source_model = model_ref
            else:
                if "species" in fields:
                    raise ValueError(
                        "'species' in fields requires all-M start "
                        "(pre-resolved config mismatch: species already resolved)."
                    )

        # Clone current state
        x_t = x_0.clone()  # keep all fields; we will update only requested ones
        x_t_dict = x_t.to_dict()

        # Static batch information
        B = len(x_0.n_atoms)
        n_atoms = x_0.n_atoms.clone()
        batch = x_0.batch.clone()
        ptr = x_0.ptr.clone()
        species = x_0.species.clone()

        # Prepare per-step storage for replay (only if requested)
        if return_trajectory:
            traj: Dict[str, torch.Tensor] = {}
            traj["times"] = times[:-1].clone()                    # [T]
            traj["dt"] = (times[1:] - times[:-1]).clone()         # [T]
            traj["n_atoms"] = n_atoms
            traj["batch"] = batch
            traj["ptr"] = ptr
            traj["species"] = species
            # initial states
            traj["pos_init"] = x_t_dict["pos"].clone()
            traj["cell_init"] = x_t_dict["cell"].clone()
            # Allocate next sequences for pos AND cell unconditionally: both fields always
            # evolve each step so the model sees on-manifold joint (pos,cell,species) states,
            # and step_logprob needs the stored next states to replay the same trajectory.
            traj["pos_next_seq"] = torch.empty(
                T, x_t_dict["pos"].shape[0], x_t_dict["pos"].shape[1], device=device, dtype=x_t_dict["pos"].dtype
            )
            traj["cell_next_seq"] = torch.empty(
                T, x_t_dict["cell"].shape[0], x_t_dict["cell"].shape[1], x_t_dict["cell"].shape[2],
                device=device, dtype=x_t_dict["cell"].dtype
            )
            # Store per-step species so replay sees the same evolved species as rollout
            traj["species_seq"] = torch.empty(
                T, species.shape[0], dtype=species.dtype, device=device
            )
        else:
            traj = {}

        # Per-field log-prob accumulators. Filled only for fields in `fields`;
        # unused channels stay zero and are omitted from the return dict.
        logp_pos_steps = torch.zeros(B, T, device=device, dtype=torch.float32)
        logp_cell_steps = torch.zeros(B, T, device=device, dtype=torch.float32)
        # Species event accumulators — one entry per atom-unmask commit.
        species_logps_list = []
        species_step_idx_list = []
        species_struct_idx_list = []

        # Convenience: per-field batch indices for log-prob aggregation
        pos_batch_idx = x_0.batch  # (sum_atoms,)

        # Main loop over time steps: one model forward per step, then update both fields
        for j in range(T):
            t = times[j]
            dt = times[j + 1] - times[j]
            sqrt_dt = torch.sqrt(dt)

            # The model expects t repeated per structure; your Model.forward handles time embedding internally.
            t_vec = t.repeat(B)
            preds = model(x_t, t_vec)  # dict-like: e.g., "pos_b","pos_eta","cell_b","cell_eta"

            # POS field: always evolve so the model sees an on-manifold joint
            # (pos, cell, species) state; accumulate logp only if "pos" is a learned field.
            si_pos = field2si["pos"]
            # Access SI parameters (epsilon, gamma, corrector, annealing)
            eps_t = si_pos._epsilon.epsilon(t)            # scalar tensor
            gam_t = si_pos._gamma.gamma(t)               # scalar tensor
            vaf = 0.0  # OMatG-IRL Sec 4.2: annealing+step count are coupled; disable for RL at N_t≤64

            corrector = si_pos._interpolant.get_corrector()

            # Predictions
            b_pos = preds["pos_b"]
            eta_pos = preds["pos_eta"]
            # Drift and diffusion
            f_pos = (1.0 + vaf * t) * b_pos - (eps_t / gam_t) * eta_pos
            g_pos = torch.sqrt(2.0 * eps_t)

            # Current and next
            x_pos = x_t_dict["pos"]
            noise = torch.randn_like(x_pos) if stochastic else torch.zeros_like(x_pos)
            proposal = x_pos + f_pos * dt + g_pos * sqrt_dt * noise
            x_pos_next = corrector.correct(proposal)

            if "pos" in fields:
                # Mean (pre-noise, corrected in the same way proposal is corrected)
                mean_pos = corrector.correct(x_pos + f_pos * dt)
                # Unwrapped delta in the correct tangent space
                x_next_unwrapped = corrector.unwrap(mean_pos, x_pos_next)
                delta = x_next_unwrapped - mean_pos  # same shape as x_pos
                # Per-atom quadratic term
                sigma2 = (g_pos ** 2) * dt
                sigma2_safe = torch.clamp(sigma2, min=1e-4)  # floor for numerical stability
                quad = (delta.pow(2).sum(dim=-1) / (sigma2_safe))  # (sum_atoms,)
                # Aggregate per-structure
                quad_sum = torch.zeros(B, device=device, dtype=torch.float32).index_add_(
                    0, pos_batch_idx, quad.to(torch.float32)
                )
                # Constant term per structure: -0.5 * d * log(2πσ^2), with d = 3 * n_atoms_i
                const = -0.5 * (3.0 * n_atoms.to(torch.float32)) * torch.log(2.0 * torch.pi * sigma2_safe.to(torch.float32))
                # Log-prob contribution
                logp_pos = -0.5 * quad_sum + const
                logp_pos_steps[:, j] = logp_pos

            # Update state and trajectory unconditionally; step_logprob needs pos_next_seq
            # to replay the same evolved states regardless of whether "pos" is learned.
            x_t_dict["pos"].copy_(x_pos_next)
            if return_trajectory:
                traj["pos_next_seq"][j].copy_(x_pos_next)

            # CELL field: always evolve (see POS comment above); gate only logp accumulation.
            si_cell = field2si["cell"]
            eps_t = si_cell._epsilon.epsilon(t)
            gam_t = si_cell._gamma.gamma(t)
            vaf = 0.0  # OMatG-IRL Sec 4.2: annealing+step count are coupled; disable for RL at N_t<=64
            corrector_c = si_cell._interpolant.get_corrector()  # usually IdentityCorrector

            b_cell = preds["cell_b"]
            eta_cell = preds["cell_eta"]
            f_cell = (1.0 + vaf * t) * b_cell - (eps_t / gam_t) * eta_cell
            g_cell = torch.sqrt(2.0 * eps_t)

            x_cell = x_t_dict["cell"]
            noise_cell = torch.randn_like(x_cell) if stochastic else torch.zeros_like(x_cell)
            proposal_cell = x_cell + f_cell * dt + g_cell * sqrt_dt * noise_cell
            x_cell_next = corrector_c.correct(proposal_cell)

            if "cell" in fields:
                mean_cell = corrector_c.correct(x_cell + f_cell * dt)
                delta_cell = (x_cell_next - mean_cell)

                sigma2_c = (g_cell ** 2) * dt
                sigma2_c_safe = torch.clamp(sigma2_c, min=1e-4)  # floor for numerical stability
                quad_cell = (delta_cell.pow(2).sum(dim=(1, 2)) / (sigma2_c_safe)).to(torch.float32)  # (B,)
                const_cell = -0.5 * (9.0) * torch.log(2.0 * torch.pi * sigma2_c_safe.to(torch.float32))  # (scalar)
                const_cell = const_cell.expand_as(quad_cell)
                logp_cell = -0.5 * quad_cell + const_cell
                logp_cell_steps[:, j] = logp_cell

            x_t_dict["cell"].copy_(x_cell_next)
            if return_trajectory:
                traj["cell_next_seq"][j].copy_(x_cell_next)


            # Record species state at this timestep for replay
            if return_trajectory:
                traj["species_seq"][j].copy_(x_t_dict["species"])

            # Species evolution: advance every step when the species started all-masked.
            # species_source_model was selected at t=0 (model if species is learned, else
            # model_ref). With resolved species the block below is skipped and species stay pinned.
            if species_starts_masked:
                # Capture PRE-evolution species and the logits used by the categorical
                # sample, so we can record per-commit event log-probs.
                pre_species = x_t_dict["species"].clone()
                captured_logits = [None]
                with torch.no_grad():
                    x_int = x_t.clone()
                    x_int_dict = x_int.to_dict()
                    def model_prediction_fn_species(time_scalar, x_species):
                        t_vec = time_scalar.repeat(len(x_t.n_atoms))
                        x_int_dict["species"].copy_(x_species)
                        preds_ref = species_source_model(x_int, t_vec)
                        logits = preds_ref["species_b"]
                        assert logits.ndim == 2, f"species logits must be 2D, got {logits.shape}"
                        assert logits.shape[0] == x_species.shape[0], \
                            f"species logits rows ({logits.shape[0]}) != atoms ({x_species.shape[0]})"
                        captured_logits[0] = logits
                        return (logits, torch.empty(0, device=logits.device))

                    x_species_next = si_species.integrate(
                        model_prediction_fn_species,
                        x_t_dict["species"], t, dt, batch,
                    )
                    x_t_dict["species"].copy_(x_species_next)

                # Extract commit events: atoms that went 0 → non-zero at step j.
                logits_j = captured_logits[0]
                post_species = x_t_dict["species"]
                committed_mask = (pre_species == 0) & (post_species != 0)
                if bool(committed_mask.any()):
                    committed_atoms = committed_mask.nonzero(as_tuple=True)[0]
                    committed_Z = post_species[committed_atoms]
                    log_probs = torch.log_softmax(logits_j[committed_atoms], dim=-1)
                    event_logp = log_probs.gather(
                        1, (committed_Z - 1).unsqueeze(1)
                    ).squeeze(1)
                    species_logps_list.append(event_logp)
                    species_step_idx_list.append(
                        torch.full(
                            (committed_atoms.numel(),), j,
                            dtype=torch.long, device=device,
                        )
                    )
                    species_struct_idx_list.append(batch[committed_atoms].to(torch.long))

        # Final state as gen
        gen = x_t

        # Assemble per-field log-prob dict. Keys are included only when the
        # corresponding channel produced meaningful numbers:
        #   - "pos"/"cell": present if the field was in `fields`.
        #   - "species":    present if species evolved (all-masked start).
        logp_dict: Dict[str, torch.Tensor] = {}
        if "pos" in fields:
            logp_dict["pos"] = logp_pos_steps
        if "cell" in fields:
            logp_dict["cell"] = logp_cell_steps
        if species_starts_masked:
            if species_logps_list:
                species_logps = torch.cat(species_logps_list, dim=0)
                species_step_idx = torch.cat(species_step_idx_list, dim=0)
                species_struct_idx = torch.cat(species_struct_idx_list, dim=0)
            else:
                species_logps = torch.zeros(0, device=device, dtype=torch.float32)
                species_step_idx = torch.zeros(0, device=device, dtype=torch.long)
                species_struct_idx = torch.zeros(0, device=device, dtype=torch.long)
            logp_dict["species"] = {
                "logp": species_logps,
                "step_idx": species_step_idx,
                "struct_idx": species_struct_idx,
            }
            # Record the final species state so step_logprob can diff at j=T-1.
            if return_trajectory:
                traj["species_final"] = x_t_dict["species"].clone()

        return gen, traj, logp_dict

    @torch.no_grad()
    def integrate_consistent(
        self,
        x_0: Data,
        model: "Model",
        fields: Tuple[str, ...] = ("pos", "cell", "species"),
        stochastic: bool = True,
    ) -> Data:
        """
        Inference/generation that uses the SAME stepper as RL training.

        Delegates to ``integrate_with_logprob`` (manual explicit Euler–Maruyama, ONE
        step per timestep, velocity-annealing vaf=0) with log-prob and trajectory
        bookkeeping disabled, and returns only the final generated state. This keeps
        train and generate on a single code path — they cannot drift — exactly as
        OMatG-IRL uses one stepper for both rollout() and eval integrate().

        The native ``integrate`` / ``_sde_integrate`` (adaptive torchsde.sdeint with
        YAML velocity annealing) is intentionally left intact for all other callers.

        :param x_0: initial points (x_0.species all-masked when species are generated, else resolved).
        :param model: the policy Model (callable returning b/eta predictions).
        :param fields: fields to integrate as SDE/DISCRETE (e.g. pos, cell, species).
        :param stochastic: if True, inject EM noise (matches RL rollout); False = mean ODE.
        :return: final generated state x_1 (Data), same batch as x_0.
        """
        gen, _, _ = self.integrate_with_logprob(
            x_0, model, fields=fields, stochastic=stochastic, return_trajectory=False,
        )
        return gen

    # next two functions add for RL
    def step_logprob(
        self,
        model: "Model",
        traj: Dict[str, torch.Tensor],
        fields: Tuple[str, ...] = ("pos", "cell"),
        return_drift: bool = False,
        return_species_dist: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Replay per-step log-probabilities under 'model' on the stored trajectory.
        Uses the same Euler–Maruyama likelihood as in integrate_with_logprob.
        Returns a per-field dict with the same shape as integrate_with_logprob.
        """
        device = traj["batch"].device
        times = traj["times"]
        dt_vec = traj["dt"]
        T = times.numel()
        B = traj["n_atoms"].shape[0]
        # Rebuild field->SI mapping to access epsilon/gamma/correctors and annealing
        field2si = {df.name: si for df, si in zip(self._data_fields, self._stochastic_interpolants)}
        # Ensure requested fields use SDE or DISCRETE (DFM species); epsilon/gamma only
        # required for SDE. Symmetric with the admission check in integrate_with_logprob.
        for fname in fields:
            si = field2si.get(fname, None)
            assert si is not None, f"Requested field '{fname}' not found among stochastic interpolants."
            de_type = getattr(si, "_differential_equation_type", None)
            assert de_type in (DifferentialEquationType.SDE, DifferentialEquationType.DISCRETE), \
                f"Field '{fname}' must use SDE or DISCRETE for RL (got {de_type})."
            if de_type == DifferentialEquationType.SDE:
                assert getattr(si, "_epsilon", None) is not None and getattr(si, "_gamma", None) is not None, \
                    f"SDE field '{fname}' must define epsilon and gamma for RL."

        # Current states (start from init)
        pos_curr = traj["pos_init"].clone()
        cell_curr = traj["cell_init"].clone()
        batch = traj["batch"]
        n_atoms = traj["n_atoms"]
        ptr = traj["ptr"]
        species_seq = traj.get("species_seq", None)  # [T, sum_atoms] or None
        species_fallback = traj["species"]  # initial species, used if species_seq not available
        species_final = traj.get("species_final", None)  # final species state (set iff species evolved)
        species_evolved = species_final is not None

        # Per-field log-prob accumulators
        logp_pos_steps = torch.zeros(B, T, device=device, dtype=torch.float32)
        logp_cell_steps = torch.zeros(B, T, device=device, dtype=torch.float32)
        species_logps_list = []
        species_step_idx_list = []
        species_struct_idx_list = []

        # Per-step full distributions at masked atoms, for the analytical categorical KL.
        species_dist_logits_list = []
        species_dist_step_idx_list = []
        species_dist_struct_idx_list = []
        species_dist_p_unmask_list = []

        # optionally collect per-step drifts for analytical KL
        if return_drift:
            drift_lists = {fname: [] for fname in fields}

        si_species = field2si.get("species", None)

        for j in range(T):
            t = times[j]
            dt = dt_vec[j]
            t_vec = t.repeat(B)

            # Use per-step species if available (matches rollout species evolution)
            species_j = species_seq[j] if species_seq is not None else species_fallback

            # Build a Data object for the current state
            x_curr = Data(
                pos=pos_curr,
                cell=cell_curr,
                species=species_j,
                n_atoms=n_atoms,
                batch=batch,
                ptr=ptr,
            )

            preds = model(x_curr, t_vec)

            if "pos" in fields:
                si_pos = field2si["pos"]
                eps_t = si_pos._epsilon.epsilon(t)
                gam_t = si_pos._gamma.gamma(t)
                vaf = 0.0  # OMatG-IRL Sec 4.2: disable annealing for RL at N_t≤64
                corrector = si_pos._interpolant.get_corrector()
                b_pos = preds["pos_b"]
                eta_pos = preds["pos_eta"]
                f_pos = (1.0 + vaf * t) * b_pos - (eps_t / gam_t) * eta_pos
                # store drift for analytical KL
                if return_drift:
                    drift_lists["pos"].append(f_pos)  # [sum_atoms, 3]
                g_pos = torch.sqrt(2.0 * eps_t)

                mean_pos = corrector.correct(pos_curr + f_pos * dt)
                next_unwrapped = corrector.unwrap(mean_pos, traj["pos_next_seq"][j])
                delta = next_unwrapped - mean_pos

                sigma2 = (g_pos ** 2) * dt
                sigma2_safe = torch.clamp(sigma2, min=1e-4) # floor to avoid blow-up
                quad = (delta.pow(2).sum(dim=-1) / (sigma2_safe))  # (sum_atoms,)
                quad_sum = torch.zeros(B, device=device, dtype=torch.float32).index_add_(0, batch, quad.to(torch.float32))
                const = -0.5 * (3.0 * n_atoms.to(torch.float32)) * torch.log(2.0 * torch.pi * sigma2_safe.to(torch.float32))
                logp_pos = -0.5 * quad_sum + const
                logp_pos_steps[:, j] = logp_pos
            # Advance pos_curr unconditionally so replay follows the same evolved state
            # sequence as the rollout (even when "pos" is not a learned field).
            pos_curr = traj["pos_next_seq"][j]

            if "cell" in fields:
                si_cell = field2si["cell"]
                eps_t = si_cell._epsilon.epsilon(t)
                gam_t = si_cell._gamma.gamma(t)
                vaf = 0.0  # OMatG-IRL Sec 4.2: disable annealing for RL at N_t<=64
                corrector_c = si_cell._interpolant.get_corrector()

                b_cell = preds["cell_b"]
                eta_cell = preds["cell_eta"]
                f_cell = (1.0 + vaf * t) * b_cell - (eps_t / gam_t) * eta_cell
                # store drift for analytical KL
                if return_drift:
                    drift_lists["cell"].append(f_cell)
                g_cell = torch.sqrt(2.0 * eps_t)

                mean_cell = corrector_c.correct(cell_curr + f_cell * dt)
                delta_cell = (traj["cell_next_seq"][j] - mean_cell)

                sigma2_c = (g_cell ** 2) * dt
                sigma2_c_safe = torch.clamp(sigma2_c, min=1e-4) # floor to avoid blow-up
                quad_cell = (delta_cell.pow(2).sum(dim=(1, 2)) / (sigma2_c_safe)).to(torch.float32)  # (B,)
                const_cell = -0.5 * (9.0) * torch.log(2.0 * torch.pi * sigma2_c_safe.to(torch.float32))
                const_cell = const_cell.expand_as(quad_cell)
                logp_cell = -0.5 * quad_cell + const_cell
                logp_cell_steps[:, j] = logp_cell
            # Advance cell_curr unconditionally so replay follows the same evolved state
            # sequence as the rollout (even when "cell" is not a learned field).
            cell_curr = traj["cell_next_seq"][j]

            # Species replay: one forward per step while any atom is still masked. The output
            # `species_b` gives both the log-probability of the unmask events and the full
            # distribution at every masked atom (for the analytical categorical KL).
            #
            # Replay must reproduce the *same* logits the rollout's species
            # sampler saw. In integrate_with_logprob the species model call
            # runs *inside* si_species.integrate, i.e. AFTER pos/cell have
            # been updated for step j; species is still pre-step-j. We
            # mirror that input state here.
            if species_evolved and si_species is not None:
                pre_species = species_seq[j]
                post_species = species_seq[j + 1] if j + 1 < T else species_final
                masked_atoms = (pre_species == 0).nonzero(as_tuple=True)[0]
                if masked_atoms.numel() > 0:
                    x_curr_species = Data(
                        pos=pos_curr,
                        cell=cell_curr,
                        species=pre_species,
                        n_atoms=n_atoms,
                        batch=batch,
                        ptr=ptr,
                    )
                    preds_species = model(x_curr_species, t_vec)
                    species_logits = preds_species["species_b"]   # [N_atoms, S]

                    # (a) Commit-event log-probs (legacy semantics, unchanged).
                    committed_mask = (pre_species == 0) & (post_species != 0)
                    if bool(committed_mask.any()):
                        committed_atoms = committed_mask.nonzero(as_tuple=True)[0]
                        committed_Z = post_species[committed_atoms]
                        log_probs = torch.log_softmax(
                            species_logits[committed_atoms], dim=-1
                        )
                        event_logp = log_probs.gather(
                            1, (committed_Z - 1).unsqueeze(1)
                        ).squeeze(1)
                        species_logps_list.append(event_logp)
                        species_step_idx_list.append(
                            torch.full(
                                (committed_atoms.numel(),), j,
                                dtype=torch.long, device=device,
                            )
                        )
                        species_struct_idx_list.append(
                            batch[committed_atoms].to(torch.long)
                        )

                    # (b) full distributions at every masked atom and the eta = 0 unmask
                    # probability p_u(t_j); computed only when the caller asks for them.
                    if return_species_dist:
                        species_dist_logits_list.append(species_logits[masked_atoms])
                        species_dist_step_idx_list.append(
                            torch.full(
                                (masked_atoms.numel(),), j,
                                dtype=torch.long, device=device,
                            )
                        )
                        species_dist_struct_idx_list.append(
                            batch[masked_atoms].to(torch.long)
                        )
                        if j == T - 1:
                            p_uj = torch.ones((), device=device, dtype=torch.float32)
                        else:
                            p_uj = (dt_vec[j] / (1.0 - times[j])).clamp(max=1.0).to(torch.float32)
                        species_dist_p_unmask_list.append(
                            p_uj.expand(masked_atoms.numel())
                        )

        # Assemble per-field log-prob dict mirroring integrate_with_logprob's return.
        logp_dict: Dict[str, torch.Tensor] = {}
        if "pos" in fields:
            logp_dict["pos"] = logp_pos_steps
        if "cell" in fields:
            logp_dict["cell"] = logp_cell_steps
        if species_evolved:
            if species_logps_list:
                species_logps = torch.cat(species_logps_list, dim=0)
                species_step_idx = torch.cat(species_step_idx_list, dim=0)
                species_struct_idx = torch.cat(species_struct_idx_list, dim=0)
            else:
                species_logps = torch.zeros(0, device=device, dtype=torch.float32)
                species_step_idx = torch.zeros(0, device=device, dtype=torch.long)
                species_struct_idx = torch.zeros(0, device=device, dtype=torch.long)
            logp_dict["species"] = {
                "logp": species_logps,
                "step_idx": species_step_idx,
                "struct_idx": species_struct_idx,
            }

        # species_dist payload (full per-atom distributions
        # at masked positions, used by _compute_analytical_kl species branch).
        if return_species_dist:
            if species_dist_logits_list:
                species_dist = {
                    "logits":     torch.cat(species_dist_logits_list, dim=0),
                    "step_idx":   torch.cat(species_dist_step_idx_list, dim=0),
                    "struct_idx": torch.cat(species_dist_struct_idx_list, dim=0),
                    "p_unmask":   torch.cat(species_dist_p_unmask_list, dim=0),
                }
            else:
                species_dist = {
                    "logits":     torch.zeros(0, MAX_ATOM_NUM, device=device, dtype=torch.float32),
                    "step_idx":   torch.zeros(0, device=device, dtype=torch.long),
                    "struct_idx": torch.zeros(0, device=device, dtype=torch.long),
                    "p_unmask":   torch.zeros(0, device=device, dtype=torch.float32),
                }

        if return_drift and return_species_dist:
            drifts = {
                fname: torch.stack(drift_lists[fname], dim=0)
                for fname in fields
                if fname in drift_lists and len(drift_lists[fname]) > 0
            }
            return logp_dict, drifts, species_dist

        if return_drift:
            # Species is DFM-discrete — no drift term, only categorical events.
            # `_compute_analytical_kl` reads the position drifts only.
            # Drop fields with empty drift_lists rather than stacking into a crash.
            drifts = {
                fname: torch.stack(drift_lists[fname], dim=0)
                for fname in fields
                if fname in drift_lists and len(drift_lists[fname]) > 0
            }
            return logp_dict, drifts

        if return_species_dist:
            return logp_dict, species_dist

        return logp_dict

    def get_stochastic_interpolant(self, data_field: str) -> StochasticInterpolant:
        """
        Return the stochastic interpolant associated with the data field.

        :param data_field:
            Data field for which the stochastic interpolant is requested.
        :type data_field: str

        :return:
            Stochastic interpolant associated with the data field.
        :rtype: StochasticInterpolant
        """
        try:
            df = DataField[data_field.lower()]
        except AttributeError:
            raise ValueError(f"Data field must be in {[d.name for d in DataField]}.")
        index = self._data_fields.index(df)
        return self._stochastic_interpolants[index]
