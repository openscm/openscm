from openscm.core.utils import hierarchical_name_as_sequence


def test_hierarchical_name_tuple_input():
    inp = ("test1", "test2")
    assert hierarchical_name_as_sequence(inp) == inp


def test_hierarchical_name_string_input():
    inp = "test1|test2"
    assert hierarchical_name_as_sequence(inp) == ["test1", "test2"]
