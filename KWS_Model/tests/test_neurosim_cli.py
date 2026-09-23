from kws.hardware.neurosim.cli import build_parser


def test_cli_exposes_only_the_initial_four_commands():
    parser = build_parser()

    assert set(parser._subparsers._group_actions[0].choices) == {
        "inspect",
        "export",
        "run",
        "validate",
    }
