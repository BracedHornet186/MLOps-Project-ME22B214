from pathlib import Path


CONFIG_ROOT_DIR = Path("conf")


def find_all_pipeline_config_files() -> list[Path]:
    return list(sorted((CONFIG_ROOT_DIR / "pipeline").glob("**/*.yaml")))


def find_pipeline_config_files(group: str, method: str) -> list[Path]:
    config_files = find_all_pipeline_config_files()
    return [
        p
        for p in config_files
        if p.is_relative_to(CONFIG_ROOT_DIR / "pipeline" / group / method)
    ]


def get_pipeline_config_file(group: str, method: str, shortname: str) -> Path:
    return CONFIG_ROOT_DIR / "pipeline" / group / method / shortname


def get_pipeline_group_names(config_files: list[Path]) -> list[str]:
    return sorted(
        list(
            set(
                [
                    str(p.relative_to(CONFIG_ROOT_DIR / "pipeline")).split("/")[0]
                    for p in config_files
                ]
            )
        )
    )


def get_pipeline_method_names(config_files: list[Path], group: str) -> list[str]:
    config_files = [p.relative_to(CONFIG_ROOT_DIR / "pipeline") for p in config_files]
    return list(
        set(
            [
                str(p.relative_to(group)).split("/")[0]
                for p in config_files
                if p.is_relative_to(group)
            ]
        )
    )


def get_pipeline_config_shortnames(
    config_files: list[Path], group: str, method: str
) -> list[str]:
    return [
        str(p.relative_to(CONFIG_ROOT_DIR / "pipeline" / group / method))
        for p in config_files
        if p.is_relative_to(CONFIG_ROOT_DIR / "pipeline" / group / method)
    ]


if __name__ == "__main__":
    ps = find_all_pipeline_config_files()
    print(ps)
    print(get_pipeline_group_names(ps))
    print(get_pipeline_method_names(ps, group="imc2024"))
