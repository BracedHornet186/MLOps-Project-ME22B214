from typing import Any, Optional
import limap.line2d
from numpy import ndarray

from scripts.data import resolve_model_path
from matchers.base import Line2DFeatureMatcher
from matchers.config import LIMAPMatcherConfig


class LIMAPMatcher(Line2DFeatureMatcher):
    def __init__(self, conf: LIMAPMatcherConfig):
        self.conf = conf

        extractor_weight_path = None
        if conf.extractor_weight_path:
            extractor_weight_path = str(resolve_model_path(conf.extractor_weight_path))
        extractor = limap.line2d.get_extractor(
            {"method": conf.extractor_method, "skip_exists": False},
            weight_path=extractor_weight_path
        )

        matcher_weight_path = None
        if conf.matcher_weight_path:
            matcher_weight_path = str(resolve_model_path(conf.matcher_weight_path))
        
        # NOTE
        # BUG
        # "n_jobs > 1" makes results random regardless of using line matching
        matcher = limap.line2d.get_matcher(
            {
                "method": conf.matcher_method,
                "skip_exists": False,
                "n_jobs": 0,
                "topk": 0,
            },
            extractor,
            weight_path=matcher_weight_path
        )
        self.matcher = matcher

    @property
    def min_matches(self) -> Optional[int]:
        return 0
    
    def match(self, descinfos1: Any, descinfos2: Any) -> ndarray:
        idxs = self.matcher.match_pair(descinfos1, descinfos2)
        #print(f"{len(idxs)} ({descinfos1[0].shape}, {descinfos2[0].shape})")
        return idxs     # type: ignore
