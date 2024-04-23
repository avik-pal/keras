import collections
import itertools
import warnings
from functools import partial

import jax
import numpy as np

from keras.src import backend
from keras.src import callbacks as callbacks_module
from keras.src import optimizers as optimizers_module
from keras.src import tree
from keras.src.backend import config
from keras.src.backend import distribution_lib as jax_distribution_lib
from keras.src.distribution import distribution_lib
from keras.src.trainers import trainer as base_trainer
from keras.src.trainers.data_adapters import array_slicing
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.epoch_iterator import EpochIterator
from keras.src.utils import traceback_utils

def enzyme(f, train):
    from enzyme_ad.jax import enzyme_jax_ir, NewXLAPipeline, OldXLAPipeline, JaXPipeline

    pipeline = JaXPipeline("""
builtin.module(
inline{default-pipeline=canonicalize max-iterations=4},
canonicalize,cse,
canonicalize,
enzyme-hlo-generate-td{
patterns=
compare_op_canon<16>;
broadcast_in_dim_op_canon<16>;
convert_op_canon<16>;
dynamic_broadcast_in_dim_op_not_actually_dynamic<16>;
chained_dynamic_broadcast_in_dim_canonicalization<16>;
dynamic_broadcast_in_dim_all_dims_non_expanding<16>;
noop_reduce_op_canon<16>;
empty_reduce_op_canon<16>;
dynamic_reshape_op_canon<16>;
get_tuple_element_op_canon<16>;
real_op_canon<16>;
imag_op_canon<16>;
get_dimension_size_op_canon<16>;
gather_op_canon<16>;
reshape_op_canon<16>;
merge_consecutive_reshapes<16>;
transpose_is_reshape<16>;
zero_extent_tensor_canon<16>;
reorder_elementwise_and_shape_op<16>;

cse_broadcast_in_dim<16>;
cse_slice<16>;
cse_transpose<16>;
cse_convert<16>;
cse_pad<16>;
cse_dot_general<16>;
cse_reshape<16>;
cse_mul<16>;
cse_div<16>;
cse_add<16>;
cse_subtract<16>;
cse_min<16>;
cse_max<16>;
cse_neg<16>;
cse_concatenate<16>;

concatenate_op_canon<16>(1024);
select_op_canon<16>(1024);
add_simplify<16>;
sub_simplify<16>;
and_simplify<16>;
max_simplify<16>;
min_simplify<16>;
or_simplify<16>;
negate_simplify<16>;
mul_simplify<16>;
div_simplify<16>;
rem_simplify<16>;
pow_simplify<16>;
sqrt_simplify<16>;
cos_simplify<16>;
sin_simplify<16>;
noop_slice<16>;
const_prop_through_barrier<16>;
slice_slice<16>;
shift_right_logical_simplify<16>;
pad_simplify<16>;
negative_pad_to_slice<16>;
tanh_simplify<16>;
exp_simplify<16>;
slice_simplify<16>;
convert_simplify<16>;
reshape_simplify<16>;
dynamic_slice_to_static<16>;
dynamic_update_slice_elim<16>;
concat_to_broadcast<16>;
reduce_to_reshape<16>;
broadcast_to_reshape<16>;
gather_simplify<16>;
iota_simplify<16>(1024);
broadcast_in_dim_simplify<16>(1024);
convert_concat<1>;
dynamic_update_to_concat<1>;
slice_of_dynamic_update<1>;
slice_elementwise<1>;
slice_pad<1>;
dot_reshape_dot<1>;
concat_const_prop<1>;
concat_fuse<1>;
pad_reshape_pad<1>;
pad_pad<1>;
concat_push_binop_add<1>;
concat_push_binop_mul<1>;
scatter_to_dynamic_update_slice<1>;
reduce_concat<1>;
slice_concat<1>;

bin_broadcast_splat_add<1>;
bin_broadcast_splat_subtract<1>;
bin_broadcast_splat_div<1>;
bin_broadcast_splat_mul<1>;
reshape_iota<16>;
slice_reshape_slice<1>;
dot_general_simplify<16>;
transpose_simplify<16>;
reshape_empty_broadcast<1>;
add_pad_pad_to_concat<1>;
broadcast_reshape<1>;

slice_reshape_concat<1>;
slice_reshape_elementwise<1>;
slice_reshape_transpose<1>;
slice_reshape_dot_general<1>;
concat_pad<1>;

reduce_pad<1>;
broadcast_pad<1>;

zero_product_reshape_pad<1>;
mul_zero_pad<1>;
div_zero_pad<1>;

binop_const_reshape_pad<1>;
binop_const_pad_add<1>;
binop_const_pad_subtract<1>;
binop_const_pad_mul<1>;
binop_const_pad_div<1>;

slice_reshape_pad<1>;
binop_binop_pad_pad_add<1>;
binop_binop_pad_pad_mul<1>;
binop_pad_pad_add<1>;
binop_pad_pad_subtract<1>;
binop_pad_pad_mul<1>;
binop_pad_pad_div<1>;
binop_pad_pad_min<1>;
binop_pad_pad_max<1>;

unary_pad_push_convert<1>;
unary_pad_push_tanh<1>;
unary_pad_push_exp<1>;

transpose_pad<1>;

transpose_dot_reorder<1>;
dot_transpose<1>;
convert_convert_float<1>;
concat_to_pad<1>;
concat_appending_reshape<1>;
reshape_iota<1>;

broadcast_reduce<1>;
slice_dot_general<1>;

dot_reshape_pad<1>;
pad_dot_general<1>(1);
pad_dot_general<1>(0);
},
transform-interpreter,
enzyme-hlo-remove-transform
)""")
    print("Enzyme pipeline!")
    if train:
            return enzyme_jax_ir(pipeline_options=pipeline, jit_options={"donate_argnums":0})(f)
    else:
            return enzyme_jax_ir(pipeline_options=pipeline)(f)

def enzyme_pre(f):
    return f
    return enzyme(f, False)

class JAXTrainer(base_trainer.Trainer):
    def __init__(self):
        super().__init__()
        self.train_function = None
        self.test_function = None
        self.predict_function = None
        self._jax_state_synced = True

    def compute_loss_and_updates(
        self,
        trainable_variables,
        non_trainable_variables,
        metrics_variables,
        x,
        y,
        sample_weight,
        training=False,
        optimizer_variables=None,
    ):
        """This method is stateless and is intended for use with jax.grad."""
        kwargs = {}
        if self._call_has_training_arg:
            kwargs["training"] = training

        # Run stateless forward pass
        y_pred, non_trainable_variables, losses = self.stateless_call(
            trainable_variables,
            non_trainable_variables,
            x,
            return_losses=True,
            **kwargs,
        )
        if losses:
            # Make forward pass losses available to compute_loss.
            self._losses_override.clear()
            self._losses_override = losses

        loss, variables = self.stateless_compute_loss(
            trainable_variables,
            non_trainable_variables,
            metrics_variables,
            x=x,
            y=y,
            y_pred=y_pred,
            sample_weight=sample_weight,
            training=training,
        )
        if losses:
            self._losses_override.clear()
        (trainable_variables, non_trainable_variables, metrics_variables) = (
            variables
        )

        # Handle loss scaling
        unscaled_loss = loss
        if training and self.optimizer is not None:
            # Scale loss with a StatelessScope, to use an update scale variable.
            mapping = list(zip(self.optimizer.variables, optimizer_variables))
            with backend.StatelessScope(state_mapping=mapping):
                loss = self.optimizer.scale_loss(loss)
        return loss, (
            unscaled_loss,
            y_pred,
            non_trainable_variables,
            metrics_variables,
        )

    def _update_metrics_variables(
        self, metrics_variables, unscaled_loss, x, y, y_pred, sample_weight
    ):
        with backend.StatelessScope(
            state_mapping=[
                (ref_v, v)
                for ref_v, v in zip(self.metrics_variables, metrics_variables)
            ]
        ) as scope:
            self._loss_tracker.update_state(
                unscaled_loss, sample_weight=tree.flatten(x)[0].shape[0]
            )
            logs = self.compute_metrics(x, y, y_pred, sample_weight)

        new_metrics_variables = []
        for ref_v in self.metrics_variables:
            new_v = scope.get_current_value(ref_v)
            if new_v is None:
                new_v = ref_v.value
            new_metrics_variables.append(new_v)
        return logs, new_metrics_variables

    def train_step(self, state, data):
        (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            metrics_variables,
        ) = state
        x, y, sample_weight = data_adapter_utils.unpack_x_y_sample_weight(data)

        loss = self.compute_loss_and_updates
        loss = enzyme_pre(loss)

        grad_fn = jax.value_and_grad(
            loss, has_aux=True
        )
        (loss, aux), grads = grad_fn(
            trainable_variables,
            non_trainable_variables,
            metrics_variables,
            x,
            y,
            sample_weight,
            training=True,
            optimizer_variables=optimizer_variables,
        )
        (unscaled_loss, y_pred, non_trainable_variables, metrics_variables) = (
            aux
        )

        (
            trainable_variables,
            optimizer_variables,
        ) = self.optimizer.stateless_apply(
            optimizer_variables, grads, trainable_variables
        )

        logs, metrics_variables = self._update_metrics_variables(
            metrics_variables, unscaled_loss, x, y, y_pred, sample_weight
        )

        state = self._enforce_jax_state_sharding(
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            metrics_variables,
        )
        return logs, state

    def test_step(self, state, data):
        (
            trainable_variables,
            non_trainable_variables,
            metrics_variables,
        ) = state
        x, y, sample_weight = data_adapter_utils.unpack_x_y_sample_weight(data)
        loss, aux = self.compute_loss_and_updates(
            trainable_variables,
            non_trainable_variables,
            metrics_variables,
            x,
            y,
            sample_weight,
            training=False,
        )
        (unscaled_loss, y_pred, non_trainable_variables, metrics_variables) = (
            aux
        )

        logs, metrics_variables = self._update_metrics_variables(
            metrics_variables, unscaled_loss, x, y, y_pred, sample_weight
        )

        (
            trainable_variables,
            non_trainable_variables,
            _,
            metrics_variables,
        ) = self._enforce_jax_state_sharding(
            trainable_variables=trainable_variables,
            non_trainable_variables=non_trainable_variables,
            optimizer_variables=None,
            metrics_variables=metrics_variables,
        )
        state = (
            trainable_variables,
            non_trainable_variables,
            metrics_variables,
        )
        return logs, state

    def predict_step(self, state, data):
        trainable_variables, non_trainable_variables = state
        kwargs = {}
        if self._call_has_training_arg:
            kwargs["training"] = False

        x, _, _ = data_adapter_utils.unpack_x_y_sample_weight(data)
        outputs, non_trainable_variables = self.stateless_call(
            trainable_variables, non_trainable_variables, x, **kwargs
        )
        (
            _,
            non_trainable_variables,
            _,
            _,
        ) = self._enforce_jax_state_sharding(
            trainable_variables=None,
            non_trainable_variables=non_trainable_variables,
            optimizer_variables=None,
            metrics_variables=None,
        )
        return outputs, non_trainable_variables

    def _make_function(self, step_function, concatenate_outputs=False):
        if self.steps_per_execution > 1:
            if concatenate_outputs:

                def concatenate(outputs):
                    output = outputs[0]
                    for next_output in outputs[1:]:
                        output = tree.map_structure(
                            lambda t1, t2: jax.numpy.concatenate([t1, t2]),
                            output,
                            next_output,
                        )
                    return output

                if not self.run_eagerly and self.jit_compile:
                    concatenate = jax.jit(concatenate)

                def iterator_step(state, iterator):
                    data = next(iterator)
                    outputs, state = step_function(state, data)
                    outputs = [outputs]
                    try:
                        for _ in range(self.steps_per_execution - 1):
                            data = next(iterator)
                            _outputs, state = step_function(state, data)
                            outputs.append(_outputs)
                    except StopIteration:
                        pass
                    outputs = concatenate(outputs)
                    return outputs, state

            else:

                def iterator_step(state, iterator):
                    data = next(iterator)
                    outputs, state = step_function(state, data)
                    try:
                        for _ in range(self.steps_per_execution - 1):
                            data = next(iterator)
                            outputs, state = step_function(state, data)
                    except StopIteration:
                        pass
                    return outputs, state

        else:

            def iterator_step(state, iterator):
                return step_function(state, next(iterator))

        return iterator_step

    def make_train_function(self, force=False):
        if self.train_function is not None and not force:
            return

        def one_train_step(state, data):
            data = data[0]
            return self.train_step(state, data)

        def multi_train_steps(state, data):
            for single_step_data in data:
                logs, state = one_train_step(state, [single_step_data])
            return logs, state

        if self.steps_per_execution > 1:
            train_step = multi_train_steps
        else:
            train_step = one_train_step
        
        if True: train_step = enzyme(train_step, True)

        if not self.run_eagerly and self.jit_compile:
            # Note that we mark the state to be donated to jax,
            # so that jax will reuse the memory buffer for outputs.
            # This will reduce the memory usage of the training function by
            # half.
            train_step = jax.jit(self.train_step, donate_argnums=0)
        else:
            train_step = self.train_step

        step_function = self._make_function(train_step)

        self.train_function = step_function

    def make_test_function(self, force=False):
        if self.test_function is not None and not force:
            return
        if not self.run_eagerly and self.jit_compile:
            # Note that we mark the state to be donated to jax,
            # so that jax will reuse the memory buffer for outputs.
            # This will reduce the memory usage of the training function by
            # half.
            test_step = jax.jit(self.test_step, donate_argnums=0)
        else:
            test_step = self.test_step

        step_function = self._make_function(test_step)

        self.test_function = step_function

    def make_predict_function(self, force=False):
        if self.predict_function is not None and not force:
            return self.predict_function

        def predict_step(state, data):
            outputs, non_trainable_variables = self.predict_step(state, data)
            return outputs, (state[0], non_trainable_variables)
        
        if True:
            predict_step = enzyme(predict_step, False)

        if not self.run_eagerly and self.jit_compile:
            predict_step = jax.jit(predict_step, donate_argnums=0)

        _step_function = self._make_function(
            predict_step, concatenate_outputs=True
        )

        def step_function(state, iterator):
            outputs, state = _step_function(state, iterator)
            return outputs, state

        self.predict_function = step_function

    @traceback_utils.filter_traceback
    def fit(
        self,
        x=None,
        y=None,
        batch_size=None,
        epochs=1,
        verbose="auto",
        callbacks=None,
        validation_split=0.0,
        validation_data=None,
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=None,
        validation_steps=None,
        validation_batch_size=None,
        validation_freq=1,
    ):
        self._assert_compile_called("fit")
        # Possibly cap epochs for debugging runs.
        max_epochs = config.max_epochs()
        if max_epochs and max_epochs < epochs:
            warnings.warn("Limiting epochs to %d" % max_epochs)
            epochs = max_epochs
        # TODO: respect compiled trainable state
        self._eval_epoch_iterator = None
        if validation_split and validation_data is None:
            # Create the validation data using the training data. Only supported
            # for TF/numpy/jax arrays.
            (
                (x, y, sample_weight),
                validation_data,
            ) = array_slicing.train_validation_split(
                (x, y, sample_weight), validation_split=validation_split
            )

        if validation_data is not None:
            (
                val_x,
                val_y,
                val_sample_weight,
            ) = data_adapter_utils.unpack_x_y_sample_weight(validation_data)

        # Create an iterator that yields batches for one epoch.
        epoch_iterator = JAXEpochIterator(
            x=x,
            y=y,
            sample_weight=sample_weight,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            shuffle=shuffle,
            class_weight=class_weight,
            steps_per_execution=self.steps_per_execution,
        )

        self._symbolic_build(iterator=epoch_iterator)
        epoch_iterator.reset()

        # Container that configures and calls callbacks.
        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(
                callbacks,
                add_history=True,
                add_progbar=verbose != 0,
                verbose=verbose,
                epochs=epochs,
                steps=epoch_iterator.num_batches,
                model=self,
            )
        self._record_training_state_sharding_spec()

        self.make_train_function()
        self.stop_training = False
        training_logs = {}
        training_finished = False
        callbacks.on_train_begin()
        initial_epoch = self._initial_epoch or initial_epoch
        try:
            for epoch in range(initial_epoch, epochs):
                self.reset_metrics()
                callbacks.on_epoch_begin(epoch)

                self._jax_state_synced = True
                with epoch_iterator.catch_stop_iteration():
                    for step, iterator in epoch_iterator:
                        # Callbacks
                        callbacks.on_train_batch_begin(step)

                        # Train step
                        if self._jax_state_synced:
                            # The state may have been synced by a callback.
                            state = self._get_jax_state(
                                trainable_variables=True,
                                non_trainable_variables=True,
                                optimizer_variables=True,
                                metrics_variables=True,
                                purge_model_variables=True,
                            )
                            self._jax_state_synced = False

                        logs, state = self.train_function(state, iterator)
                        (
                            trainable_variables,
                            non_trainable_variables,
                            optimizer_variables,
                            metrics_variables,
                        ) = state

                        # Setting _jax_state enables callbacks to force a state
                        # sync if they need to.
                        self._jax_state = {
                            "trainable_variables": trainable_variables,
                            "non_trainable_variables": non_trainable_variables,
                            "optimizer_variables": optimizer_variables,
                            "metrics_variables": metrics_variables,
                        }
                        # Dispatch callbacks. This takes care of async dispatch.
                        callbacks.on_train_batch_end(step, logs)

                        if self.stop_training:
                            # Stop training if a callback has set
                            # this flag in on_(train_)batch_end.
                            break

                # Reattach state to the model
                # (if not already done by a callback).
                # NOTE: doing this after each step would be a big performance
                # bottleneck.
                self.jax_state_sync()

                # Override with model metrics instead of last step logs if
                # needed.
                epoch_logs = dict(self._get_metrics_result_or_logs(logs))

                # Run validation.
                if validation_data is not None and self._should_eval(
                    epoch, validation_freq
                ):
                    # Create JAXEpochIterator for evaluation and cache it.
                    if getattr(self, "_eval_epoch_iterator", None) is None:
                        self._eval_epoch_iterator = JAXEpochIterator(
                            x=val_x,
                            y=val_y,
                            sample_weight=val_sample_weight,
                            batch_size=validation_batch_size or batch_size,
                            steps_per_execution=self.steps_per_execution,
                            steps_per_epoch=validation_steps,
                            shuffle=False,
                        )
                    val_logs = self.evaluate(
                        x=val_x,
                        y=val_y,
                        sample_weight=val_sample_weight,
                        batch_size=validation_batch_size or batch_size,
                        steps=validation_steps,
                        callbacks=callbacks,
                        return_dict=True,
                        _use_cached_eval_dataset=True,
                    )
                    val_logs = {
                        "val_" + name: val for name, val in val_logs.items()
                    }
                    epoch_logs.update(val_logs)

                callbacks.on_epoch_end(epoch, epoch_logs)
                training_logs = epoch_logs
                if self.stop_training:
                    break
            training_finished = True

        finally:
            self.jax_state_sync()
            if (
                isinstance(self.optimizer, optimizers_module.Optimizer)
                and epochs > 0
            ):
                self.optimizer.finalize_variable_values(self.trainable_weights)

            # If _eval_epoch_iterator exists, delete it after all epochs
            # are done.
            if getattr(self, "_eval_epoch_iterator", None) is not None:
                del self._eval_epoch_iterator
            if training_finished:
                callbacks.on_train_end(logs=training_logs)
            self._jax_state = None
            self._clear_jax_state_sharding()
        return self.history

    @traceback_utils.filter_traceback
    def evaluate(
        self,
        x=None,
        y=None,
        batch_size=None,
        verbose="auto",
        sample_weight=None,
        steps=None,
        callbacks=None,
        return_dict=False,
        **kwargs,
    ):
        self._assert_compile_called("evaluate")
        # TODO: respect compiled trainable state
        use_cached_eval_dataset = kwargs.pop("_use_cached_eval_dataset", False)
        if kwargs:
            raise ValueError(f"Arguments not recognized: {kwargs}")

        if use_cached_eval_dataset:
            epoch_iterator = self._eval_epoch_iterator
        else:
            # Create an iterator that yields batches of
            # input/target data.
            epoch_iterator = JAXEpochIterator(
                x=x,
                y=y,
                sample_weight=sample_weight,
                batch_size=batch_size,
                steps_per_epoch=steps,
                shuffle=False,
                steps_per_execution=self.steps_per_execution,
            )

        self._symbolic_build(iterator=epoch_iterator)
        epoch_iterator.reset()

        # Container that configures and calls callbacks.
        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(
                callbacks,
                add_progbar=verbose != 0,
                verbose=verbose,
                epochs=1,
                steps=epoch_iterator.num_batches,
                model=self,
            )
        self._record_training_state_sharding_spec()

        self.make_test_function()
        self.stop_evaluating = False
        callbacks.on_test_begin()
        logs = {}
        self.reset_metrics()

        self._jax_state_synced = True
        with epoch_iterator.catch_stop_iteration():
            for step, iterator in epoch_iterator:
                callbacks.on_test_batch_begin(step)

                if self._jax_state_synced:
                    # The state may have been synced by a callback.
                    state = self._get_jax_state(
                        trainable_variables=True,
                        non_trainable_variables=True,
                        metrics_variables=True,
                        purge_model_variables=True,
                    )
                    self._jax_state_synced = False

                logs, state = self.test_function(state, iterator)
                (
                    trainable_variables,
                    non_trainable_variables,
                    metrics_variables,
                ) = state

                # Setting _jax_state enables callbacks to force a state sync
                # if they need to.
                self._jax_state = {
                    # I wouldn't recommend modifying non-trainable model state
                    # during evaluate(), but it's allowed.
                    "trainable_variables": trainable_variables,
                    "non_trainable_variables": non_trainable_variables,
                    "metrics_variables": metrics_variables,
                }

                # Dispatch callbacks. This takes care of async dispatch.
                callbacks.on_test_batch_end(step, logs)

                if self.stop_evaluating:
                    break

        # Reattach state back to model (if not already done by a callback).
        self.jax_state_sync()

        logs = self._get_metrics_result_or_logs(logs)
        callbacks.on_test_end(logs)
        self._jax_state = None
        if not use_cached_eval_dataset:
            # Only clear sharding if evaluate is not called from `fit`.
            self._clear_jax_state_sharding()
        if return_dict:
            return logs
        return self._flatten_metrics_in_order(logs)

    @traceback_utils.filter_traceback
    def predict(
        self, x, batch_size=None, verbose="auto", steps=None, callbacks=None
    ):
        # Create an iterator that yields batches of input data.
        epoch_iterator = JAXEpochIterator(
            x=x,
            batch_size=batch_size,
            steps_per_epoch=steps,
            shuffle=False,
            steps_per_execution=self.steps_per_execution,
        )

        if not all(layer.built for layer in self._flatten_layers()):
            # Build the model on one batch of data.
            for _, iterator in epoch_iterator:
                # Build model
                x, _, _ = data_adapter_utils.unpack_x_y_sample_weight(
                    next(iterator)
                )
                with backend.StatelessScope():
                    self(x)
                break
            epoch_iterator.reset()
        # Container that configures and calls callbacks.
        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(
                callbacks,
                add_progbar=verbose != 0,
                verbose=verbose,
                epochs=1,
                steps=epoch_iterator.num_batches,
                model=self,
            )
        self._record_training_state_sharding_spec()

        self.make_predict_function()
        self.stop_predicting = False
        callbacks.on_predict_begin()

        def append_to_outputs(batch_outputs, outputs):
            if outputs is None:
                outputs = tree.map_structure(
                    lambda batch_output: [batch_output],
                    batch_outputs,
                )
            else:
                tree.map_structure_up_to(
                    batch_outputs,
                    lambda output, batch_output: output.append(batch_output),
                    outputs,
                    batch_outputs,
                )
            return outputs

        self._jax_state_synced = True
        outputs = None
        non_trainable_variables = None
        with epoch_iterator.catch_stop_iteration():
            for step, iterator in epoch_iterator:
                callbacks.on_predict_batch_begin(step)
                if self._jax_state_synced:
                    # The state may have been synced by a callback.
                    state = self._get_jax_state(
                        trainable_variables=True,
                        non_trainable_variables=True,
                        purge_model_variables=True,
                    )
                    self._jax_state_synced = False
                batch_outputs, state = self.predict_function(state, iterator)
                (
                    trainable_variables,
                    non_trainable_variables,
                ) = state
                self._jax_state = {
                    "trainable_variables": trainable_variables,
                    # I wouldn't recommend modifying non-trainable model state
                    # during predict(), but it's allowed.
                    "non_trainable_variables": non_trainable_variables,
                }
                outputs = append_to_outputs(batch_outputs, outputs)

                # Dispatch callbacks. This takes care of async dispatch.
                callbacks.on_predict_batch_end(step, {"outputs": batch_outputs})

                if self.stop_predicting:
                    break

        self.jax_state_sync()
        callbacks.on_predict_end()
        self._jax_state = None
        self._clear_jax_state_sharding()
        return tree.map_structure_up_to(batch_outputs, np.concatenate, outputs)

    def train_on_batch(
        self,
        x,
        y=None,
        sample_weight=None,
        class_weight=None,
        return_dict=False,
    ):
        self._assert_compile_called("train_on_batch")
        if class_weight is not None:
            if sample_weight is not None:
                raise ValueError(
                    "Arguments `sample_weight` and `class_weight` "
                    "cannot be specified at the same time. "
                    f"Received: sample_weight={sample_weight}, "
                    f"class_weight={class_weight}"
                )
            sample_weight = data_adapter_utils.class_weight_to_sample_weights(
                y, class_weight
            )

        def data():
            yield _distribute_data((x, y, sample_weight))

        # Maybe build model
        self._symbolic_build(data_batch=next(data()))
        self._record_training_state_sharding_spec()
        self.make_train_function()

        # Train step
        state = self._get_jax_state(
            trainable_variables=True,
            non_trainable_variables=True,
            optimizer_variables=True,
            metrics_variables=True,
            purge_model_variables=False,
        )
        self._jax_state_synced = False
        logs, state = self.train_function(state, data())

        # State sync
        (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            metrics_variables,
        ) = state
        self._jax_state = {
            "trainable_variables": trainable_variables,
            "non_trainable_variables": non_trainable_variables,
            "optimizer_variables": optimizer_variables,
            "metrics_variables": metrics_variables,
        }
        self.jax_state_sync()

        # Format return values
        logs = tree.map_structure(lambda x: np.array(x), logs)
        if return_dict:
            return logs
        return self._flatten_metrics_in_order(logs)

    def test_on_batch(
        self,
        x,
        y=None,
        sample_weight=None,
        return_dict=False,
    ):
        self._assert_compile_called("test_on_batch")

        def data():
            yield _distribute_data((x, y, sample_weight))

        # Maybe build model
        self._symbolic_build(data_batch=next(data()))
        self._record_training_state_sharding_spec()
        self.make_test_function()

        # Test step
        state = self._get_jax_state(
            trainable_variables=True,
            non_trainable_variables=True,
            metrics_variables=True,
            purge_model_variables=False,
        )
        self._jax_state_synced = False
        logs, state = self.test_function(state, data())

        # State sync
        trainable_variables, non_trainable_variables, metrics_variables = state
        self._jax_state = {
            "trainable_variables": trainable_variables,
            "non_trainable_variables": non_trainable_variables,
            "metrics_variables": metrics_variables,
        }
        self.jax_state_sync()

        # Format return values.
        logs = tree.map_structure(lambda x: np.array(x), logs)
        if return_dict:
            return logs
        return self._flatten_metrics_in_order(logs)

    def predict_on_batch(self, x):
        if not all(layer.built for layer in self._flatten_layers()):
            # Build model
            with backend.StatelessScope():
                self(x)
        self._record_training_state_sharding_spec()
        self.make_predict_function()

        state = self._get_jax_state(
            trainable_variables=True,
            non_trainable_variables=True,
            metrics_variables=False,
            purge_model_variables=False,
        )
        self._jax_state_synced = False

        def data():
            yield (x,)

        batch_outputs, state = self.predict_function(state, data())
        trainable_variables, non_trainable_variables = state
        self._jax_state = {
            "trainable_variables": trainable_variables,
            "non_trainable_variables": non_trainable_variables,
        }
        self.jax_state_sync()
        batch_outputs = tree.map_structure(lambda x: np.array(x), batch_outputs)
        return batch_outputs

    def jax_state_sync(self):
        if not getattr(self, "_jax_state", None) or self._jax_state_synced:
            return

        trainable_variables = self._jax_state.get("trainable_variables", None)
        non_trainable_variables = self._jax_state.get(
            "non_trainable_variables", None
        )
        optimizer_variables = self._jax_state.get("optimizer_variables", None)
        metrics_variables = self._jax_state.get("metrics_variables", None)
        if trainable_variables:
            for ref_v, v in zip(self.trainable_variables, trainable_variables):
                ref_v.assign(v)
        if non_trainable_variables:
            for ref_v, v in zip(
                self.non_trainable_variables, non_trainable_variables
            ):
                ref_v.assign(v)
        if optimizer_variables:
            for ref_v, v in zip(self.optimizer.variables, optimizer_variables):
                ref_v.assign(v)
        if metrics_variables:
            for ref_v, v in zip(self.metrics_variables, metrics_variables):
                ref_v.assign(v)
        self._jax_state_synced = True

    def _record_training_state_sharding_spec(self):
        self._trainable_variable_shardings = [
            v.value.sharding for v in self.trainable_variables
        ]
        self._non_trainable_variable_shardings = [
            v.value.sharding for v in self.non_trainable_variables
        ]
        if hasattr(self, "optimizer") and self.optimizer is not None:
            self._optimizer_variable_shardings = [
                v.value.sharding for v in self.optimizer.variables
            ]
        else:
            self._optimizer_variable_shardings = []
        self._metrics_variable_shardings = [
            v.value.sharding for v in self.metrics_variables
        ]

    def _clear_jax_state_sharding(self):
        self._trainable_variable_shardings = None
        self._non_trainable_variable_shardings = None
        self._optimizer_variable_shardings = None
        self._metrics_variable_shardings = None

    def _enforce_jax_state_sharding(
        self,
        trainable_variables=None,
        non_trainable_variables=None,
        optimizer_variables=None,
        metrics_variables=None,
    ):
        """Enforce the sharding spec constraint for all the training state.

        Since the output of the train/eval step will be used as inputs to next
        step, we need to ensure that they have the same sharding spec, so that
        jax.jit won't have to recompile the train/eval function.

        Note that this function will also rely on the recorded sharding spec
        for each of states.

        This function is expected to be called within the jitted train/eval
        function, especially around the end of the function.
        """
        trainable_variables = trainable_variables or []
        non_trainable_variables = non_trainable_variables or []
        optimizer_variables = optimizer_variables or []
        metrics_variables = metrics_variables or []

        for i in range(len(trainable_variables)):
            trainable_variables[i] = jax.lax.with_sharding_constraint(
                trainable_variables[i], self._trainable_variable_shardings[i]
            )
        for i in range(len(non_trainable_variables)):
            non_trainable_variables[i] = jax.lax.with_sharding_constraint(
                non_trainable_variables[i],
                self._non_trainable_variable_shardings[i],
            )
        for i in range(len(optimizer_variables)):
            optimizer_variables[i] = jax.lax.with_sharding_constraint(
                optimizer_variables[i], self._optimizer_variable_shardings[i]
            )
        for i in range(len(metrics_variables)):
            metrics_variables[i] = jax.lax.with_sharding_constraint(
                metrics_variables[i], self._metrics_variable_shardings[i]
            )
        return (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            metrics_variables,
        )

    def _purge_model_variables(
        self,
        trainable_variables=False,
        non_trainable_variables=False,
        optimizer_variables=False,
        metrics_variables=False,
    ):
        """Remove all the model variable for memory saving.

        During JAX training, since the training function is stateless, we have
        to pass in and get the model weights over and over, during which the
        copy of the weights that attached to the Variable are still and
        occupying extra memory. We remove those variable to save memory (for
        better memory utilization) at the beginning of the epoch, and reattach
        the value back to variables at the end of the epoch, via
        `jax_state_sync()`.
        """
        if trainable_variables:
            for v in self.trainable_variables:
                v._value = None
        if non_trainable_variables:
            for v in self.non_trainable_variables:
                v._value = None
        if optimizer_variables:
            for v in self.optimizer.variables:
                v._value = None
        if metrics_variables:
            for v in self.metrics_variables:
                v._value = None

    def _get_jax_state(
        self,
        trainable_variables=False,
        non_trainable_variables=False,
        optimizer_variables=False,
        metrics_variables=False,
        purge_model_variables=False,
    ):
        state = []
        if trainable_variables:
            state.append([v.value for v in self.trainable_variables])
        if non_trainable_variables:
            state.append([v.value for v in self.non_trainable_variables])
        if optimizer_variables:
            state.append([v.value for v in self.optimizer.variables])
        if metrics_variables:
            state.append([v.value for v in self.metrics_variables])
        if purge_model_variables:
            self._purge_model_variables(
                trainable_variables=trainable_variables,
                non_trainable_variables=non_trainable_variables,
                optimizer_variables=optimizer_variables,
                metrics_variables=metrics_variables,
            )
        return tuple(state)


def _distribute_data(data, layouts=None):
    distribution = distribution_lib.distribution()

    if distribution is not None:
        if layouts is None:
            layouts = tree.map_structure(
                lambda d: distribution.get_data_layout(d.shape),
                data,
            )
        jax_dist_data_input = partial(
            jax_distribution_lib.distribute_data_input,
            batch_dim_name=distribution.batch_dim_name,
        )
        return tree.map_structure(jax_dist_data_input, data, layouts)

    return tree.map_structure(jax.device_put, data)


class JAXEpochIterator(EpochIterator):
    def __next__(self):
        return next(self._epoch_iterator)

    def _get_iterator(self):
        distribution = distribution_lib.distribution()
        if distribution is not None:
            return self._get_distributed_iterator(distribution)

        return self._prefetch_numpy_iterator(
            self.data_adapter.get_jax_iterator()
        )

    def _get_distributed_iterator(self, distribution):
        """Lazily compute layouts to reduce host to device transfer latency."""
        layouts = None
        for data in self.data_adapter.get_jax_iterator():
            if layouts is None:
                layouts = tree.map_structure(
                    lambda d: distribution.get_data_layout(
                        d.shape
                    ).backend_layout,
                    data,
                )
            yield _distribute_data(data, layouts)

    def _prefetch_numpy_iterator(self, numpy_iterator):
        """Shard and prefetch batches on device.

        Most of the implementation has been borrowed from
        `flax.jax_utils.prefetch_to_device`

        This utility takes an iterator and returns a new iterator which fills an
        on device prefetch buffer. Eager prefetching can improve the performance
        of training loops significantly by overlapping compute and data
        transfer.
        """
        queue = collections.deque()

        # If you're training on GPUs, 2 is generally the best choice because
        # this guarantees that you can overlap a training step on GPU with a
        # data prefetch step on CPU.
        def enqueue(n=2):
            for data in itertools.islice(numpy_iterator, n):
                queue.append(_distribute_data(data))

        enqueue(n=2)  # TODO: should we make `n` configurable?
        while queue:
            yield queue.popleft()
            enqueue(1)
