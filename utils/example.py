from __future__ import annotations

from typing import Self

from data import DEFAULT_DATASET_DIR, DataSchema, IMC2025TrainData


class ExampleScene:
    dataset_name: str

    @classmethod
    def to_schema(cls: type[Self]) -> DataSchema:
        return IMC2025TrainData.create(
            DEFAULT_DATASET_DIR, datasets_to_use=[cls.dataset_name]
        )


class Church(ExampleScene):
    dataset_name = "imc2023_theather_imc2024_church"

    easy_positive_pair_01 = (
        DEFAULT_DATASET_DIR / "train/imc2023_theather_imc2024_church/church_00092.png",
        DEFAULT_DATASET_DIR / "train/imc2023_theather_imc2024_church/church_00099.png",
    )

    easy_positive_pair_02 = (
        DEFAULT_DATASET_DIR / "train/imc2023_theather_imc2024_church/church_00016.png",
        DEFAULT_DATASET_DIR / "train/imc2023_theather_imc2024_church/church_00017.png",
    )

    hard_positive_pair_01 = (
        DEFAULT_DATASET_DIR / "train/imc2023_theather_imc2024_church/church_00040.png",
        DEFAULT_DATASET_DIR / "train/imc2023_theather_imc2024_church/church_00043.png",
    )


class ET(ExampleScene):
    dataset_name = "ETs"

    easy_positive_pair_01 = (
        DEFAULT_DATASET_DIR / "train/ETs/another_et_another_et001.png",
        DEFAULT_DATASET_DIR / "train/ETs/another_et_another_et002.png",
    )

    hard_positive_pair_01 = (
        DEFAULT_DATASET_DIR / "train/ETs/another_et_another_et001.png",
        DEFAULT_DATASET_DIR / "train/ETs/another_et_another_et010.png",
    )

    hard_positive_pair_02 = (
        DEFAULT_DATASET_DIR / "train/ETs/et_et001.png",
        DEFAULT_DATASET_DIR / "train/ETs/et_et008.png",
    )

    easy_negative_pair_01 = (
        DEFAULT_DATASET_DIR / "train/ETs/another_et_another_et001.png",
        DEFAULT_DATASET_DIR / "train/ETs/et_et000.png",
    )


class Haiper(ExampleScene):
    dataset_name = "imc2023_haiper"

    easy_positive_pair_01 = (
        DEFAULT_DATASET_DIR / "train/imc2023_haiper/bike_image_029.png",
        DEFAULT_DATASET_DIR / "train/imc2023_haiper/bike_image_038.png",
    )

    easy_negative_pair_01 = (
        DEFAULT_DATASET_DIR / "train/imc2023_haiper/fountain_image_041.png",
        DEFAULT_DATASET_DIR / "train/imc2023_haiper/bike_image_038.png",
    )


class Dioscuri(ExampleScene):
    dataset_name = "imc2024_dioscuri_baalshamin"

    easy_positive_pair_01 = (
        DEFAULT_DATASET_DIR
        / "train/imc2024_dioscuri_baalshamin/baalshamin_dscn2032.png",
        DEFAULT_DATASET_DIR
        / "train/imc2024_dioscuri_baalshamin/baalshamin_dscn2034.png",
    )


class Stairs(ExampleScene):
    dataset_name = "stairs"

    easy_positive_pair_01 = (
        DEFAULT_DATASET_DIR / "train/stairs/stairs_split_1_1710453576271.png",
        DEFAULT_DATASET_DIR / "train/stairs/stairs_split_1_1710453601885.png",
    )

    hard_positive_pair_01 = (
        DEFAULT_DATASET_DIR / "train/stairs/stairs_split_1_1710453576271.png",
        DEFAULT_DATASET_DIR / "train/stairs/stairs_split_1_1710453606287.png",
    )

    hard_positive_pair_02 = (
        DEFAULT_DATASET_DIR / "train/stairs/stairs_split_1_1710453668718.png",
        DEFAULT_DATASET_DIR / "train/stairs/stairs_split_1_1710453616892.png",
    )

    easy_negative_pair_01 = (
        DEFAULT_DATASET_DIR / "train/stairs/stairs_split_1_1710453576271.png",
        DEFAULT_DATASET_DIR / "train/stairs/stairs_split_1_1710453651110.png",
    )

    easy_negative_pair_02 = (
        DEFAULT_DATASET_DIR / "train/stairs/stairs_split_1_1710453678922.png",
        DEFAULT_DATASET_DIR / "train/stairs/stairs_split_1_1710453947066.png",
    )

    easy_negative_pair_03 = (
        DEFAULT_DATASET_DIR / "train/stairs/stairs_split_2_1710453790978.png",
        DEFAULT_DATASET_DIR / "train/stairs/stairs_split_2_1710453725143.png",
    )


class Theather(ExampleScene):
    dataset_name = "imc2023_theather_imc2024_church"

    hard_positive_pair_01 = (
        DEFAULT_DATASET_DIR
        / "train/imc2023_theather_imc2024_church/kyiv_puppet_theater_img_20220127_170350.png",
        DEFAULT_DATASET_DIR
        / "train/imc2023_theather_imc2024_church/kyiv_puppet_theater_img_20220127_170633.png",
    )
