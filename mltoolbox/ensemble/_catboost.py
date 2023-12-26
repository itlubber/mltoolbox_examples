from catboost import CatBoostClassifier, CatBoostRegressor

from ..base import ModelParameterProxy


def has_catboost():
    try:
        import catboost
        return True
    except ImportError:
        return False


def isa_catboost_model(clf):
    if not has_catboost():
        return False
    from catboost import CatBoost # noqa: F401
    return isinstance(clf, CatBoost)


def is_catboost_model(cls):
    if not has_catboost():
        return False
    from catboost import CatBoost # noqa: F401
    return issubclass(cls, CatBoost)


class CatBoostClassifierParameterProxy(ModelParameterProxy):
    def __init__(self, iterations=None, learning_rate=None, depth=None, l2_leaf_reg=None, rsm=None, model_size_reg=None,
                 loss_function=None, border_count=None, feature_border_type=None, input_borders=None, output_borders=None,
                 fold_permutation_block=None, od_pval=None, od_wait=None, od_type=None, nan_mode=None, counter_calc_method=None,
                 leaf_estimation_iterations=None, leaf_estimation_method=None, thread_count=None, random_seed=None,
                 use_best_model=None, best_model_min_trees=None, verbose=None, silent=None, logging_level=None, metric_period=None,
                 ctr_leaf_count_limit=None, store_all_simple_ctr=None, max_ctr_complexity=None, has_time=None, allow_const_label=None,
                 target_border=None, classes_count=None, class_weights=None, class_names=None, one_hot_max_size=None,
                 random_strength=None, name=None, ignored_features=None, train_dir=None, custom_loss=None, custom_metric=None,
                 eval_metric=None, bagging_temperature=None, save_snapshot=None, snapshot_file=None, snapshot_interval=None,
                 fold_len_multiplier=None, used_ram_limit=None, gpu_ram_part=None, pinned_memory_size=None, allow_writing_files=None,
                 final_ctr_computation_mode=None, approx_on_full_history=None, boosting_type=None, simple_ctr=None,
                 combinations_ctr=None, per_feature_ctr=None, ctr_description=None, ctr_target_border_count=None, task_type=None,
                 device_config=None, devices=None, bootstrap_type=None, subsample=None, sampling_unit=None, 
                 dev_score_calc_obj_block_size=None, dev_efb_max_buckets=None, efb_max_conflict_fraction=None, max_depth=None,
                 n_estimators=None, num_boost_round=None, num_trees=None, colsample_bylevel=None, random_state=None,
                 reg_lambda=None, objective=None, eta=None, max_bin=None, scale_pos_weight=None, gpu_cat_features_storage=None,
                 data_partition=None, metadata=None, early_stopping_rounds=None, cat_features=None, grow_policy=None,
                 min_data_in_leaf=None, max_leaves=None, score_function=None, leaf_estimation_backtracking=None, ctr_history_unit=None):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.rsm = rsm
        self.loss_function = loss_function
        self.border_count = border_count
        self.feature_border_type = feature_border_type
        self.input_borders = input_borders
        self.output_borders = output_borders
        self.fold_permutation_block = fold_permutation_block
        self.od_pval = od_pval
        self.od_wait = od_wait
        self.od_type = od_type
        self.nan_mode = nan_mode
        self.counter_calc_method = counter_calc_method
        self.leaf_estimation_iterations = leaf_estimation_iterations
        self.leaf_estimation_method = leaf_estimation_method
        self.thread_count = thread_count
        self.random_seed = random_seed
        self.use_best_model = use_best_model
        self.best_model_min_trees = best_model_min_trees
        self.verbose = verbose
        self.silent = silent
        self.logging_level = logging_level
        self.metric_period = metric_period
        self.ctr_leaf_count_limit = ctr_leaf_count_limit
        self.store_all_simple_ctr = store_all_simple_ctr
        self.max_ctr_complexity = max_ctr_complexity
        self.has_time = has_time
        self.allow_const_label = allow_const_label
        self.target_border = target_border
        self.classes_count = classes_count
        self.class_weights = class_weights
        self.class_names = class_names
        self.one_hot_max_size = one_hot_max_size
        self.random_strength = random_strength
        self.name = name
        self.ignored_features = ignored_features
        self.train_dir = train_dir
        self.custom_loss = custom_loss
        self.custom_metric = custom_metric
        self.eval_metric = eval_metric
        self.bagging_temperature = bagging_temperature
        self.save_snapshot = save_snapshot
        self.snapshot_file = snapshot_file
        self.snapshot_interval = snapshot_interval
        self.fold_len_multiplier = fold_len_multiplier
        self.used_ram_limit = used_ram_limit
        self.gpu_ram_part = gpu_ram_part
        self.pinned_memory_size = pinned_memory_size
        self.allow_writing_files = allow_writing_files
        self.final_ctr_computation_mode = final_ctr_computation_mode
        self.approx_on_full_history = approx_on_full_history
        self.boosting_type = boosting_type
        self.simple_ctr = simple_ctr
        self.model_size_reg = model_size_reg
        self.combinations_ctr = combinations_ctr
        self.per_feature_ctr = per_feature_ctr
        self.ctr_description = ctr_description
        self.ctr_target_border_count = ctr_target_border_count
        self.task_type = task_type
        self.device_config = device_config
        self.devices = devices
        self.bootstrap_type = bootstrap_type
        self.subsample = subsample
        self.sampling_unit = sampling_unit
        self.dev_score_calc_obj_block_size = dev_score_calc_obj_block_size
        self.dev_efb_max_buckets = dev_efb_max_buckets
        self.efb_max_conflict_fraction = efb_max_conflict_fraction
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.num_boost_round = num_boost_round
        self.num_trees = num_trees
        self.colsample_bylevel = colsample_bylevel
        self.random_state = random_state
        self.reg_lambda = reg_lambda
        self.objective = objective
        self.eta = eta
        self.max_bin = max_bin
        self.scale_pos_weight = scale_pos_weight
        self.gpu_cat_features_storage = gpu_cat_features_storage
        self.data_partition = data_partition
        self.early_stopping_rounds = early_stopping_rounds
        self.metadata = metadata
        self.cat_features = cat_features
        self.grow_policy = grow_policy
        self.min_data_in_leaf = min_data_in_leaf
        self.max_leaves = max_leaves
        self.score_function = score_function
        self.leaf_estimation_backtracking = leaf_estimation_backtracking
        self.ctr_history_unit = ctr_history_unit

    def _make_estimator(self):
        return CatBoostClassifier(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            model_size_reg=self.model_size_reg,
            rsm=self.rsm,
            loss_function=self.loss_function,
            border_count=self.border_count,
            feature_border_type=self.feature_border_type,
            input_borders=self.input_borders,
            output_borders=self.output_borders,
            fold_permutation_block=self.fold_permutation_block,
            od_pval=self.od_pval,
            od_wait=self.od_wait,
            od_type=self.od_type,
            nan_mode=self.nan_mode,
            counter_calc_method=self.counter_calc_method,
            leaf_estimation_iterations=self.leaf_estimation_iterations,
            leaf_estimation_method=self.leaf_estimation_method,
            thread_count=self.thread_count,
            random_seed=self.random_seed,
            use_best_model=self.use_best_model,
            best_model_min_trees=self.best_model_min_trees,
            verbose=self.verbose,
            silent=self.silent,
            logging_level=self.logging_level,
            metric_period=self.metric_period,
            ctr_leaf_count_limit=self.ctr_leaf_count_limit,
            store_all_simple_ctr=self.store_all_simple_ctr,
            max_ctr_complexity=self.max_ctr_complexity,
            has_time=self.has_time,
            target_border=self.target_border,
            classes_count=self.classes_count,
            class_weights=self.class_weights,
            class_names=self.class_names,
            one_hot_max_size=self.one_hot_max_size,
            random_strength=self.random_strength,
            name=self.name,
            ignored_features=self.ignored_features,
            train_dir=self.train_dir,
            custom_loss=self.custom_loss,
            custom_metric=self.custom_metric,
            eval_metric=self.eval_metric,
            bagging_temperature=self.bagging_temperature,
            save_snapshot=self.save_snapshot,
            snapshot_file=self.snapshot_file,
            snapshot_interval=self.snapshot_interval,
            fold_len_multiplier=self.fold_len_multiplier,
            used_ram_limit=self.used_ram_limit,
            gpu_ram_part=self.gpu_ram_part,
            pinned_memory_size=self.pinned_memory_size,
            allow_writing_files=self.allow_writing_files,
            final_ctr_computation_mode=self.final_ctr_computation_mode,
            approx_on_full_history=self.approx_on_full_history,
            boosting_type=self.boosting_type,
            simple_ctr=self.simple_ctr,
            combinations_ctr=self.combinations_ctr,
            per_feature_ctr=self.per_feature_ctr,
            ctr_description=self.ctr_description,
            ctr_target_border_count=self.ctr_target_border_count,
            task_type=self.task_type,
            device_config=self.device_config,
            devices=self.devices,
            bootstrap_type=self.bootstrap_type,
            subsample=self.subsample,
            sampling_unit=self.sampling_unit,
            dev_score_calc_obj_block_size=self.dev_score_calc_obj_block_size,
            dev_efb_max_buckets=self.dev_efb_max_buckets,
            efb_max_conflict_fraction=self.efb_max_conflict_fraction,
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            allow_const_label=self.allow_const_label,
            num_boost_round=self.num_boost_round,
            num_trees=self.num_trees,
            colsample_bylevel=self.colsample_bylevel,
            random_state=self.random_state,
            objective=self.objective,
            reg_lambda=self.reg_lambda,
            eta=self.eta,
            max_bin=self.max_bin,
            scale_pos_weight=self.scale_pos_weight,
            gpu_cat_features_storage=self.gpu_cat_features_storage,
            data_partition=self.data_partition,
            metadata=self.metadata,
            early_stopping_rounds=self.early_stopping_rounds,
            cat_features=self.cat_features,
            grow_policy=self.grow_policy,
            min_data_in_leaf=self.min_data_in_leaf,
            max_leaves=self.max_leaves,
            score_function=self.score_function,
            leaf_estimation_backtracking=self.leaf_estimation_backtracking,
            ctr_history_unit=self.ctr_history_unit
        )


class CatBoostRegressorParameterProxy(ModelParameterProxy):
    def __init__(self, iterations=None, learning_rate=None, depth=None, l2_leaf_reg=None, model_size_reg=None, rsm=None,
                 loss_function='RMSE', border_count=None, feature_border_type=None, input_borders=None, output_borders=None,
                 fold_permutation_block=None, od_pval=None, od_wait=None, od_type=None, nan_mode=None, counter_calc_method=None,
                 leaf_estimation_iterations=None, leaf_estimation_method=None, thread_count=None, random_seed=None,
                 use_best_model=None, best_model_min_trees=None, verbose=None, silent=None, logging_level=None, metric_period=None,
                 ctr_leaf_count_limit=None, store_all_simple_ctr=None, max_ctr_complexity=None, has_time=None, allow_const_label=None,
                 target_border=None, classes_count=None, class_weights=None, class_names=None, one_hot_max_size=None,
                 random_strength=None, name=None, ignored_features=None, train_dir=None, custom_loss=None, custom_metric=None,
                 eval_metric=None, bagging_temperature=None, save_snapshot=None, snapshot_file=None, snapshot_interval=None,
                 fold_len_multiplier=None, used_ram_limit=None, gpu_ram_part=None, pinned_memory_size=None, allow_writing_files=None,
                 final_ctr_computation_mode=None, approx_on_full_history=None, boosting_type=None, simple_ctr=None,
                 combinations_ctr=None, per_feature_ctr=None, ctr_description=None, ctr_target_border_count=None, task_type=None,
                 device_config=None, devices=None, bootstrap_type=None, subsample=None, sampling_unit=None, monotone_constraints=None,
                 dev_score_calc_obj_block_size=None, dev_efb_max_buckets=None, efb_max_conflict_fraction=None, max_depth=None,
                 n_estimators=None, num_boost_round=None, num_trees=None, colsample_bylevel=None, random_state=None,
                 reg_lambda=None, objective=None, eta=None, max_bin=None, scale_pos_weight=None, gpu_cat_features_storage=None,
                 data_partition=None, metadata=None, early_stopping_rounds=None, cat_features=None, grow_policy=None,
                 min_data_in_leaf=None, max_leaves=None, score_function=None, leaf_estimation_backtracking=None, ctr_history_unit=None,):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.model_size_reg = model_size_reg
        self.rsm = rsm
        self.loss_function = loss_function
        self.border_count = border_count
        self.feature_border_type = feature_border_type
        self.input_borders = input_borders
        self.output_borders = output_borders
        self.fold_permutation_block = fold_permutation_block
        self.od_pval = od_pval
        self.od_wait = od_wait
        self.od_type = od_type
        self.nan_mode = nan_mode
        self.counter_calc_method = counter_calc_method
        self.leaf_estimation_iterations = leaf_estimation_iterations
        self.leaf_estimation_method = leaf_estimation_method
        self.thread_count = thread_count
        self.random_seed = random_seed
        self.use_best_model = use_best_model
        self.silent = silent
        self.save_snapshot = save_snapshot
        self.snapshot_file = snapshot_file
        self.snapshot_interval = snapshot_interval
        self.used_ram_limit = used_ram_limit
        self.gpu_ram_part = gpu_ram_part
        self.pinned_memory_size = pinned_memory_size
        self.allow_writing_files = allow_writing_files
        self.final_ctr_computation_mode = final_ctr_computation_mode
        self.approx_on_full_history = approx_on_full_history
        self.boosting_type = boosting_type
        self.simple_ctr = simple_ctr
        self.combinations_ctr = combinations_ctr
        self.per_feature_ctr = per_feature_ctr
        self.ctr_description = ctr_description
        self.best_model_min_trees = best_model_min_trees
        self.verbose = verbose
        self.logging_level = logging_level
        self.metric_period = metric_period
        self.ctr_leaf_count_limit = ctr_leaf_count_limit
        self.store_all_simple_ctr = store_all_simple_ctr
        self.max_ctr_complexity = max_ctr_complexity
        self.has_time = has_time
        self.allow_const_label = allow_const_label
        self.target_border = target_border
        self.one_hot_max_size = one_hot_max_size
        self.random_strength = random_strength
        self.name = name
        self.ignored_features = ignored_features
        self.train_dir = train_dir
        self.custom_metric = custom_metric
        self.eval_metric = eval_metric
        self.bagging_temperature = bagging_temperature
        self.fold_len_multiplier = fold_len_multiplier
        self.ctr_target_border_count = ctr_target_border_count
        self.task_type = task_type
        self.device_config = device_config
        self.devices = devices
        self.bootstrap_type = bootstrap_type
        self.subsample = subsample
        self.sampling_unit = sampling_unit
        self.dev_score_calc_obj_block_size = dev_score_calc_obj_block_size
        self.dev_efb_max_buckets = dev_efb_max_buckets
        self.efb_max_conflict_fraction = efb_max_conflict_fraction
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.num_boost_round = num_boost_round
        self.colsample_bylevel = colsample_bylevel
        self.num_trees = num_trees
        self.random_state = random_state
        self.reg_lambda = reg_lambda
        self.objective = objective
        self.eta = eta
        self.max_bin = max_bin
        self.gpu_cat_features_storage = gpu_cat_features_storage
        self.data_partition = data_partition
        self.metadata = metadata
        self.early_stopping_rounds = early_stopping_rounds
        self.cat_features = cat_features
        self.grow_policy = grow_policy
        self.max_leaves = max_leaves
        self.min_data_in_leaf = min_data_in_leaf
        self.score_function = score_function
        self.leaf_estimation_backtracking = leaf_estimation_backtracking
        self.ctr_history_unit = ctr_history_unit
        self.monotone_constraints = monotone_constraints

    def _make_estimator(self):
        return CatBoostRegressor(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            model_size_reg=self. model_size_reg,
            rsm=self.rsm,
            loss_function=self.loss_function,
            border_count=self.border_count,
            feature_border_type=self.feature_border_type,
            input_borders=self.input_borders,
            output_borders=self.output_borders,
            fold_permutation_block=self.fold_permutation_block,
            od_pval=self.od_pval,
            od_wait=self.od_wait,
            od_type=self.od_type,
            nan_mode=self.nan_mode,
            counter_calc_method=self.counter_calc_method,
            leaf_estimation_iterations=self.leaf_estimation_iterations,
            leaf_estimation_method=self.leaf_estimation_method,
            thread_count=self.thread_count,
            random_seed=self.random_seed,
            use_best_model=self.use_best_model,
            best_model_min_trees=self.best_model_min_trees,
            verbose=self.verbose,
            silent=self.silent,
            logging_level=self.logging_level,
            metric_period=self.metric_period,
            ctr_leaf_count_limit=self.ctr_leaf_count_limit,
            store_all_simple_ctr=self.store_all_simple_ctr,
            max_ctr_complexity=self.max_ctr_complexity,
            has_time=self.has_time,
            allow_const_label=self.allow_const_label,
            target_border=self.target_border,
            one_hot_max_size=self.one_hot_max_size,
            random_strength=self.random_strength,
            name=self.name,
            ignored_features=self.ignored_features,
            train_dir=self.train_dir,
            custom_metric=self.custom_metric,
            eval_metric=self.eval_metric,
            bagging_temperature=self.bagging_temperature,
            save_snapshot=self.save_snapshot,
            snapshot_file=self.snapshot_file,
            snapshot_interval=self.snapshot_interval,
            dev_efb_max_buckets=self.dev_efb_max_buckets,
            fold_len_multiplier=self.fold_len_multiplier,
            used_ram_limit=self.used_ram_limit,
            gpu_ram_part=self.gpu_ram_part,
            pinned_memory_size=self.pinned_memory_size,
            allow_writing_files=self.allow_writing_files,
            approx_on_full_history=self.approx_on_full_history,
            boosting_type=self.boosting_type,
            final_ctr_computation_mode=self.final_ctr_computation_mode,
            simple_ctr=self.simple_ctr,
            combinations_ctr=self.combinations_ctr,
            per_feature_ctr=self.per_feature_ctr,
            ctr_description=self.ctr_description,
            ctr_target_border_count=self.ctr_target_border_count,
            task_type=self.task_type,
            device_config=self.device_config,
            devices=self.devices,
            bootstrap_type=self.bootstrap_type,
            subsample=self.subsample,
            sampling_unit=self.sampling_unit,
            dev_score_calc_obj_block_size=self.dev_score_calc_obj_block_size,
            efb_max_conflict_fraction=self.efb_max_conflict_fraction,
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            num_boost_round=self.num_boost_round,
            num_trees=self.num_trees,
            colsample_bylevel=self.colsample_bylevel,
            random_state=self.random_state,
            reg_lambda=self.reg_lambda,
            objective=self.objective,
            eta=self.eta,
            max_bin=self.max_bin,
            gpu_cat_features_storage=self.gpu_cat_features_storage,
            data_partition=self.data_partition,
            metadata=self.metadata,
            early_stopping_rounds=self.early_stopping_rounds,
            cat_features=self.cat_features,
            grow_policy=self.grow_policy,
            min_data_in_leaf=self.min_data_in_leaf,
            max_leaves=self.max_leaves,
            score_function=self.score_function,
            leaf_estimation_backtracking=self.leaf_estimation_backtracking,
            ctr_history_unit=self.ctr_history_unit,
            monotone_constraints=self.monotone_constraints
        )
