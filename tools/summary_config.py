from argparse import ArgumentParser, Namespace

from config import (
    DetectorBasedPipelineConfig,
    DetectorFreePipelineConfig,
    PipelineConfig,
    PreMatchingEnsemblePipelineConfig,
    SimpleEnsemblePipelineConfig,
    IMC2024PipelineConfig
)
from features.config import LocalFeatureConfig
from matchers.config import DetectorFreeMatcherConfig, LocalFeatureMatcherConfig


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("conf")
    return parser.parse_args()


def _lfm_format(m: LocalFeatureMatcherConfig) -> str:
    return str(getattr(m, m.type))


def _dfm_format(m: DetectorFreeMatcherConfig) -> str:
    return f"{str(getattr(m, m.type))}[mkpts={m.mkpts_decoupling_method}, cropper={m.cropper_type}]"


def _localfeature_format(c: LocalFeatureConfig) -> str:
    return str(getattr(c, c.type))


def get_local_features(conf: PipelineConfig) -> str:
    c = conf.get_core_config()
    if isinstance(c, DetectorFreePipelineConfig):
        return ""
    return ", ".join([_localfeature_format(f) for f in c.local_features])


def get_line2d(conf: PipelineConfig) -> str:
    c = conf.get_core_config()
    if hasattr(c, "line2d_features"):
        assert c.line2d_features is not None
        return ", ".join([str(f) for f in c.line2d_features])
    return ""


def get_matchers(conf: PipelineConfig) -> str:
    c = conf.get_core_config()

    if isinstance(c, DetectorBasedPipelineConfig):
        return _lfm_format(c.matcher)
    elif isinstance(c, DetectorFreePipelineConfig):
        return _dfm_format(c.matcher)
    elif isinstance(
        c, (PreMatchingEnsemblePipelineConfig, SimpleEnsemblePipelineConfig, IMC2024PipelineConfig)
    ):
        return ", ".join(
            [_lfm_format(m) for m in c.local_feature_matchers]
            + [_dfm_format(m) for m in c.detector_free_matchers]
        )
    raise TypeError


def get_pre_matcher(conf: PipelineConfig) -> str:
    c = conf.get_core_config()
    if isinstance(c, PreMatchingEnsemblePipelineConfig):
        if c.pre_matcher:
            return _dfm_format(c.pre_matcher)
        else:
            return "None"
    elif isinstance(c, IMC2024PipelineConfig):
        if c.pre_matching:
            return str(c.pre_matching)
        return "None"
    return "None"


def get_shortlist(conf: PipelineConfig) -> str:
    c = conf.get_core_config()
    return str(c.shortlist_generator)


def get_shortlist_updater(conf: PipelineConfig) -> str:
    c = conf.get_core_config()
    if isinstance(c, (PreMatchingEnsemblePipelineConfig, IMC2024PipelineConfig,)):
        if c.shortlist_updater:
            return str(c.shortlist_updater)
        return ''
    return ''


def get_resize(conf: PipelineConfig) -> str:
    c = conf.get_core_config()
    resize_configs = []
    if isinstance(c, DetectorFreePipelineConfig):
        if c.matcher.loftr:
            assert c.matcher.loftr.resize
            kwargs = {
                "func": c.matcher.loftr.resize.func,
                "small": c.matcher.loftr.resize.small_edge_length,
                "antialias": c.matcher.loftr.resize.antialias,
            }
        else:
            return ""
        func = kwargs.pop("func")
        param_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
        resize_str = f"{func}({param_str})"
        return resize_str

    for lf in c.local_features:
        resize_configs.append(lf.resize)
    resize_strs = []
    for rc in resize_configs:
        if rc is None:
            resize_strs.append("")
            continue

        kwargs = {}
        if rc.lg_resize:
            kwargs["size"] = rc.lg_resize
        elif rc.ml_resize:
            kwargs["size"] = rc.ml_resize
        elif rc.long_edge_length:
            kwargs["long"] = rc.long_edge_length
        elif rc.small_edge_length:
            kwargs["small"] = rc.small_edge_length
        elif rc.size:
            kwargs["size"] = rc.size

        param_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
        resize_strs.append(f"{rc.func}({param_str})")
    return ", ".join(resize_strs)


def get_oe(conf: PipelineConfig) -> str:
    c = conf.get_core_config()
    if isinstance(c, (PreMatchingEnsemblePipelineConfig, IMC2024PipelineConfig)):
        return str(c.overlap_region_estimation)
    return ""


def get_deblurring(conf: PipelineConfig) -> str:
    c = conf.get_core_config()
    if hasattr(c, 'deblurring'):
        if c.deblurring is None:
            return ""
        return str(c.deblurring)
    return ""


def get_segmentation(conf: PipelineConfig) -> str:
    c = conf.get_core_config()
    if hasattr(c, 'segmentation'):
        if c.segmentation is None:
            return ""
        return str(c.segmentation)
    return ""


def get_ori(conf: PipelineConfig) -> str:
    c = conf.get_core_config()
    if hasattr(c, 'orientation_normalization'):
        if c.orientation_normalization is None:
            return ""
        return str(c.orientation_normalization)
    return ""


def get_masking(conf: PipelineConfig) -> str:
    c = conf.get_core_config()
    if isinstance(c, (IMC2024PipelineConfig,)):
        if c.masking is None:
            return ""
        return str(c.masking)
    return ""


def get_gv(conf: PipelineConfig) -> str:
    c = conf.get_core_config()
    return str(c.verification)


def get_pruning(conf: PipelineConfig) -> str:
    c = conf.get_core_config()
    if isinstance(c, IMC2024PipelineConfig):
        if not c.pruning:
            return ""
        return str(c.pruning)
    return ""


def get_hloc(conf: PipelineConfig) -> str:
    c = conf.get_core_config()
    if isinstance(c, IMC2024PipelineConfig):
        if not c.hloc_match_dense:
            return ""
        return str(c.hloc_match_dense)
    return ""


def get_rec(conf: PipelineConfig) -> str:
    c = conf.get_core_config()
    return str(c.reconstruction)


def main():
    args = parse_args()

    conf = PipelineConfig.load_config(args.conf)

    columns = []
    columns.append(("Pipeline Config", args.conf))
    columns.append(("Kernel", ""))
    columns.append(("Kernel Version", ""))
    columns.append(("Public LB", ""))
    columns.append(("Eval", ""))
    columns.append(("Local (church)", ""))
    columns.append(("Local (dioscuri)", ""))
    columns.append(("Local (lizard)", ""))
    columns.append(("Local (mttb)", ""))
    columns.append(("Local (pond)", ""))
    columns.append(("Local (cup)", ""))
    columns.append(("Local (cylinder)", ""))
    columns.append(("Pipeline", conf.type))
    columns.append(("LocalFeatures", get_local_features(conf)))
    columns.append(("Matchers", get_matchers(conf)))
    columns.append(("Line2D", get_line2d(conf)))
    columns.append(("PreMatcher", get_pre_matcher(conf)))
    columns.append(("Shortlist", get_shortlist(conf)))
    columns.append(("ShortlistUpdater", get_shortlist_updater(conf)))
    columns.append(("Resize", get_resize(conf)))
    columns.append(("Segmentation", get_segmentation(conf)))
    columns.append(("OverlapEstimation", get_oe(conf)))
    columns.append(("Deblurring", get_deblurring(conf)))
    columns.append(("Orientation", get_ori(conf)))
    columns.append(("Masking", get_masking(conf)))
    columns.append(("GV", get_gv(conf)))
    columns.append(("Pruning", get_pruning(conf)))
    columns.append(("HLoc", get_hloc(conf)))
    columns.append(("Reconstruction", get_rec(conf)))

    _, values = zip(*columns)
    print("\t".join(values))


if __name__ == "__main__":
    main()
