import sys
from pathlib import Path
path_root1 = Path(__file__).parents[1]
sys.path.append(str(path_root1))

import numpy as np
import os
from numpy import testing as tst
import pytest
from spectraljet import Components, PDGNames
from .tools import generic_equality_comp, TempTestDir, data_dir
from .micro_samples import AwkdArrays
import awkward as ak


def test_flatten():
    input_ouputs = [
            (AwkdArrays.empty, []),
            (AwkdArrays.one_one, [1]),
            (AwkdArrays.minus_plus, range(-2, 3)),
            (AwkdArrays.event_ints, [1,2,3]),
            (AwkdArrays.jet_ints, list(range(1, 7))),
            (AwkdArrays.event_floats, [.1, .2, .3]),
            (AwkdArrays.jet_floats, np.arange(0.1, 0.7, 0.1)),
            (AwkdArrays.empty_event, [1]),
            (AwkdArrays.empty_jet, [1,2,3]),
            (AwkdArrays.empty_events, []),
            (AwkdArrays.empty_jets, [])]
    for inp, out in input_ouputs:
        inp = list(Components.flatten(inp))
        tst.assert_allclose(inp, out)


def test_detect_depth():
    input_ouputs = [
            (AwkdArrays.empty, (False, 0)),
            (AwkdArrays.one_one, (True, 0)),
            (AwkdArrays.minus_plus, (True, 0)),
            (AwkdArrays.event_ints, (True, 1)),
            (AwkdArrays.jet_ints, (True, 2)),
            (AwkdArrays.event_floats, (True, 1)),
            (AwkdArrays.jet_floats, (True, 2)),
            (AwkdArrays.empty_event, (True, 1)),
            (AwkdArrays.empty_jet, (True, 2)),
            (AwkdArrays.empty_events, (False, 1)),
            (AwkdArrays.empty_jets, (False, 2))]
    for inp, out in input_ouputs:
        depth = Components.detect_depth(inp)
        assert depth == out, f"{inp} gives depth (certanty, depth) = {depth}, not {out}"


def test_apply_array_func():
    func = np.cos
    input_ouputs = [
        (AwkdArrays.empty, ak.from_iter([])),
        (AwkdArrays.one_one, ak.from_iter([func(1)])),
        (AwkdArrays.minus_plus, ak.from_iter(func(np.arange(-2, 3)))),
        (AwkdArrays.event_ints, ak.from_iter([[func(1), func(2)], [func(3)]])),
        (AwkdArrays.jet_ints, ak.from_iter([ak.from_iter([[func(1), func(2)], [func(3)]]),
                                     ak.from_iter([[func(4),func(5),func(6)]])])),
        (AwkdArrays.event_floats, ak.from_iter([[func(.1), func(.2)], [func(.3)]])),
        (AwkdArrays.jet_floats, ak.from_iter([ak.from_iter([[func(.1), func(.2)], [func(.3)]]),
                                     ak.from_iter([[func(.4),func(.5),func(.6)]])])),
        (AwkdArrays.empty_event, ak.from_iter([[], [func(1)]])),
        (AwkdArrays.empty_jet, ak.from_iter([ak.from_iter([[], [func(1)]]),
                                      ak.from_iter([[func(2), func(3)]])])),
        (AwkdArrays.empty_events, ak.from_iter([[], []])),
        (AwkdArrays.empty_jets, ak.from_iter([ak.from_iter([[],[]]),
                                       ak.from_iter([[]])]))]
    for inp, out in input_ouputs:
        result = Components.apply_array_func(func, inp)
        assert generic_equality_comp(out, result), f"{inp} gives result = {result}, not {out}"
    func = len
    input_ouputs = [
        (AwkdArrays.empty, 0),
        (AwkdArrays.one_one, 1),
        (AwkdArrays.minus_plus, 5),
        (AwkdArrays.event_ints, ak.from_iter([2, 1])),
        (AwkdArrays.jet_ints, ak.from_iter([ak.from_iter([2, 1]),
                                     ak.from_iter([3])])),
        (AwkdArrays.event_floats, ak.from_iter([2, 1])),
        (AwkdArrays.jet_floats, ak.from_iter([ak.from_iter([2, 1]),
                                     ak.from_iter([3])])),
        (AwkdArrays.empty_event, ak.from_iter([0, 1])),
        (AwkdArrays.empty_jet, ak.from_iter([ak.from_iter([0, 1]),
                                      ak.from_iter([2])])),
        (AwkdArrays.empty_events, ak.from_iter([0, 0])),
        (AwkdArrays.empty_jets, ak.from_iter([[0, 0], [0]]))]
    for inp, out in input_ouputs:
        result = Components.apply_array_func(func, inp)
        assert generic_equality_comp(out, result), f"{inp} gives result = {result}, not {out}"
    # specify event depth
    input_ouputs = [
        (AwkdArrays.event_ints, ak.from_iter([2, 1])),
        (AwkdArrays.jet_ints, ak.from_iter([2, 1])),
        (AwkdArrays.event_floats, ak.from_iter([2, 1])),
        (AwkdArrays.jet_floats, ak.from_iter([2, 1])),
        (AwkdArrays.empty_event, ak.from_iter([0, 1])),
        (AwkdArrays.empty_jet, ak.from_iter([2, 1])),
        (AwkdArrays.empty_events, ak.from_iter([0, 0])),
        (AwkdArrays.empty_jets, ak.from_iter([2, 1]))]
    for inp, out in input_ouputs:
        result = Components.apply_array_func(func, inp, depth=Components.EventWise.EVENT_DEPTH)
        assert generic_equality_comp(out, result), f"{inp} gives result = {result}, not {out}"


def test_confine_angle():
    inputs_outputs = [
            (0., 0.),
            (1., 1.),
            (-1., -1.),
            (2*np.pi, 0.),
            (np.pi+0.1, -np.pi+0.1),
            (-np.pi-0.1, np.pi-0.1),
            (2*np.pi+0.1, 0.1),
            (-2*np.pi, 0.),
            (-2*np.pi-0.1, -0.1),
            (4*np.pi, 0.)]
    for inp, out in inputs_outputs:
        tst.assert_allclose(Components.confine_angle(inp), out)


class TestAngularDistance:
    def function(self, a, b):
        return Components.angular_distance(a, b)

    def test_angular_distance(self):
        inputs_a = np.array([0., 0., 0., 0., -1., -np.pi-0.1])
        inputs_b = np.array([0., 1., -1., 2*np.pi, 1., np.pi+0.1])
        expected = np.array([0., 1., 1., 0., 2., 0.2])
        found = Components.angular_distance(inputs_a, inputs_b)
        tst.assert_allclose(found, expected)
        # check giving a single out
        single_out = Components.angular_distance(0., 0.)
        assert single_out == 0.


def test_raw_to_angular_distance():
    inputs = np.array([0., np.pi, -np.pi, 2*np.pi, 3*np.pi])
    expected = np.array([0., np.pi, np.pi, 0,  np.pi])
    found = Components.raw_to_angular_distance(inputs)
    tst.assert_allclose(found, expected)
    # check a single output
    single_out = Components.raw_to_angular_distance(0.)
    assert single_out == 0


def test_safe_convert():
    # edge case tests
    examples = (('None', str, None),
                ('None', int, None),
                ('None', float, None),
                ('None', bool, None),
                ('1', str, '1'),
                ('1', int, 1),
                ('1', float, 1.),
                ('1', bool, True),
                ('0', str, '0'),
                ('0', int, 0),
                ('0', float, 0.),
                ('0', bool, False),
                ('-1', str, '-1'),
                ('-1', int, -1),
                ('-1', float, -1.),
                ('-1', bool, True),
                ('0.5', str, '0.5'),
                ('0.5', float, 0.5),
                ('0.5', bool, True))
    for inp, cls, exptd in examples:
        out = Components.safe_convert(cls, inp)
        assert out == exptd, "Components.safe_convert failed to convert " +\
                             f"{inp} to {exptd} via {cls} " +\
                             f"instead got {out}."
    # try some random numbers
    for num in np.random.uniform(-1000, 1000, 20):
        inp = str(num)
        cls = float
        exptd = num
        out = Components.safe_convert(cls, inp)
        assert out == exptd, "Components.safe_convert failed to convert " +\
                             f"{inp} to {exptd} via {cls} " +\
                             f"instead got {out}."
        cls = bool
        exptd = True
        out = Components.safe_convert(cls, inp)
        assert out == exptd, "Components.safe_convert failed to convert " +\
                             f"{inp} to {exptd} via {cls} " +\
                             f"instead got {out}."
        cls = int
        exptd = int(num)
        out = Components.safe_convert(cls, inp.split('.', 1)[0])
        assert out == exptd, "Components.safe_convert failed to convert " +\
                             f"{inp} to {exptd} via {cls} " +\
                             f"instead got {out}."


def test_EventWise():
    with TempTestDir("tst") as dir_name:
        # instansation
        file_name = "blank.parquet"
        blank_ew = Components.EventWise(os.path.join(dir_name, file_name))
        assert blank_ew.columns == []
        # getting attributes
        with pytest.raises(AttributeError):
            getattr(blank_ew, "PT")
        with pytest.raises(AttributeError):
            blank_ew.PT
        # write
        save_path = os.path.join(dir_name, file_name)
        blank_ew.write()
        assert os.path.exists(save_path)
        # from file
        blank_ew_clone = Components.EventWise.from_file(save_path)
        assert generic_equality_comp(blank_ew.columns, blank_ew_clone.columns)
        contents = {k: blank_ew._column_contents[k].to_list() for k in
                    ak.fields(blank_ew._column_contents)
                    if k not in ["column_order",
                                 "hyperparameter_column_order"]
                    and "gitdict" not in k}
        contents_clone = {k: blank_ew_clone._column_contents[k].to_list() for k in
                          ak.fields(blank_ew_clone._column_contents)
                          if k not in ["column_order",
                                       "hyperparameter_column_order"]
                          and "gitdict" not in k}
        assert generic_equality_comp(contents, contents_clone)
        # eq
        assert blank_ew == blank_ew_clone
        # append
        blank_ew.append(**{"A": AwkdArrays.empty})
        blank_ew.append(B=AwkdArrays.one_one)
        assert list(blank_ew.A) == []
        assert "A" in blank_ew.columns
        assert list(blank_ew.B) == [1]
        assert "B" in blank_ew.columns
        # remove
        current_cols = list(blank_ew.columns)
        for name in current_cols:
            blank_ew.remove(name)
        assert len(blank_ew.columns) == 0
        with pytest.raises(AttributeError):
            blank_ew.a
        with pytest.raises(AttributeError):
            blank_ew.b
        # remove prefix
        blank_ew.append(**{"A": AwkdArrays.empty, "Bc": AwkdArrays.one_one, "Bd": AwkdArrays.minus_plus})
        blank_ew.remove_prefix("B")
        assert list(blank_ew.columns) == ["A"]
        # cannot remove somehting that does not exist
        try:
            blank_ew.remove("non_existant")
            raise AssertionError("Should throw an error when removing columns that don't exist")
        except KeyError:
            pass
        # instancate with content
        content = {"A1": AwkdArrays.minus_plus, "Long_name": AwkdArrays.minus_plus, "Hyper": AwkdArrays.one_one, "Prefix_1": AwkdArrays.jet_ints, "Prefix_2": AwkdArrays.jet_floats}
        columns = ["A1", "Long_name", "Prefix_1", "Prefix_2"]
        hyperparameter_columns = ["Hyper"]
        filled_name = "filled.parquet"
        alt_ew = Components.EventWise(os.path.join(dir_name, filled_name),
                                      columns=columns,
                                      contents=content,
                                      hyperparameter_columns=hyperparameter_columns)
        assert list(alt_ew.columns) == columns
        assert list(alt_ew.hyperparameter_columns) == hyperparameter_columns
        assert generic_equality_comp(alt_ew.Hyper, AwkdArrays.one_one)
        # make an alias
        alt_ew.add_alias("A2", "A1")
        assert generic_equality_comp(alt_ew.A2, AwkdArrays.minus_plus)
        # read and write with the alias
        alt_ew.write()
        alt_ew_clone = Components.EventWise.from_file(os.path.join(dir_name, filled_name))
        assert list(alt_ew_clone.columns) == columns
        assert list(alt_ew_clone.hyperparameter_columns) == hyperparameter_columns
        assert generic_equality_comp(alt_ew_clone.Hyper, AwkdArrays.one_one)
        assert generic_equality_comp(alt_ew_clone.A2, AwkdArrays.minus_plus)
        # check the dir contains both parameters and hyperparameters
        dir_cont = alt_ew_clone.__dir__()
        assert "A2" in dir_cont
        for key in columns + hyperparameter_columns + ["A2"]:
            assert key in dir_cont
        # remove an alias
        alt_ew_clone.remove("A2")
        assert "A2" not in alt_ew_clone.__dir__()
        assert "A2" not in alt_ew_clone.columns
        assert generic_equality_comp(alt_ew_clone.A1, AwkdArrays.minus_plus)
        # overwrite a hyperparameter
        alt_ew_clone.append_hyperparameters(Hyper=AwkdArrays.one_one*2)
        assert generic_equality_comp(alt_ew_clone.Hyper, 2*alt_ew.Hyper)
        # fail to overwrite a hyperparameter with a parameter
        try:
            alt_ew_clone.append(Hyper=AwkdArrays.one_one)
            raise AssertionError("Should not be possible to overwrite a column wiht a hayperparamter")
        except KeyError:
            pass
        # fail to overwrite a parameter with a hyperparamter
        try:
            alt_ew_clone.append_hyperparameters(A1=AwkdArrays.one_one)
            raise AssertionError("Should not be possible to overwrite a hyperparameter with a paramter")
        except KeyError:
            pass
        # rename a column
        alt_ew.rename("Prefix_1", "Prefix_A")
        assert generic_equality_comp(alt_ew.Prefix_A, AwkdArrays.jet_ints)
        assert "Prefix_1" not in alt_ew.columns
        # rename a prefix
        alt_ew.rename_prefix("Prefix", "Dog")
        assert generic_equality_comp(alt_ew.Dog_A, AwkdArrays.jet_ints)
        assert generic_equality_comp(alt_ew.Dog_2, AwkdArrays.jet_floats)
        # rename a hyperparameter
        alt_ew.rename("Hyper", "Hyperdog")
        assert generic_equality_comp(alt_ew.Hyperdog, AwkdArrays.one_one)
        assert "Hyper" not in alt_ew.hyperparameter_columns
        # rename an alias
        alt_ew.rename("A2", "Alias2")
        assert "A2" not in alt_ew.columns
        assert generic_equality_comp(alt_ew.Alias2, AwkdArrays.minus_plus)
        # cannot rename something we don't have
        try:
            alt_ew.rename("non_existant", "sphinx")
            raise AssertionError("Shouldn't be able to rename an non-existant column")
        except KeyError:
            pass


# test subsections of eventwise ~~~~~~~~~~~~~~~

def test_split():
    with TempTestDir("tst") as dir_name:
        # splitting a blank ew should result in only Nones
        file_name = "test.parquet"
        ew = Components.EventWise(os.path.join(dir_name, file_name))
        ew.append(Energy= ak.from_iter([]))
        parts = ew.split([0, 0, 0], [0, 0, 0])
        for part in parts:
            assert part is None
        # otherwise things should be possible to divide into events
        # try with 10 events
        n_events = 10
        content_1 = ak.from_iter(np.arange(n_events))
        content_2 = ak.from_iter(np.random.rand(n_events))
        content_3 = ak.from_iter([np.random.rand(np.random.randint(5)) for _ in range(n_events)])
        content_4 = ak.from_iter([[ak.from_iter(np.random.rand(np.random.randint(5)))
                                       for _ in range(np.random.randint(5))]
                                      for _ in range(n_events)])
        ew.append(c1=content_1, c2=content_2, c3=content_3, c4=content_4)
        # check nothing changes in the original
        tst.assert_allclose(ew.c1, content_1)
        tst.assert_allclose(ew.c2, content_2)
        tst.assert_allclose(ak.flatten(ew.c3), ak.flatten(content_3))
        tst.assert_allclose(ak.flatten(ak.flatten(ew.c4)),
                            ak.flatten(ak.flatten(content_4)))
        paths = ew.split([0, 5, 7, 7], [5, 7, 7, 10], "c1", "dog")
        tst.assert_allclose(ew.c1, content_1)
        tst.assert_allclose(ew.c2, content_2)
        tst.assert_allclose(ak.flatten(ew.c3), ak.flatten(content_3))
        tst.assert_allclose(ak.flatten(ak.flatten(ew.c4)),
                            ak.flatten(ak.flatten(content_4)))
        # check the segments contain what they should
        ew0 = Components.EventWise.from_file(paths[0])
        assert len(ew0.c1) == 5
        tst.assert_allclose(ew0.c1, content_1[:5])
        tst.assert_allclose(ew0.c2, content_2[:5])
        tst.assert_allclose(ak.flatten(ew0.c3), ak.flatten(content_3[:5]))
        tst.assert_allclose(ak.flatten(ak.flatten(ew0.c4)),
                            ak.flatten(ak.flatten(content_4[:5])))
        ew1 = Components.EventWise.from_file(paths[1])
        assert len(ew1.c1) == 2
        tst.assert_allclose(ew1.c1, content_1[5:7])
        tst.assert_allclose(ew1.c2, content_2[5:7])
        tst.assert_allclose(ak.flatten(ew1.c3), ak.flatten(content_3[5:7]))
        
        flat_ew = ak.flatten(ew1.c4)
        flat_content = ak.flatten(content_4[5:7])
        try:
            flat_content = ak.flatten(flat_content)
            flat_ew = ak.flatten(flat_ew)
        except ValueError:  # already flat
            pass
        tst.assert_allclose(flat_ew, flat_content)
        
        assert paths[2] is None
        ew3 = Components.EventWise.from_file(paths[3])
        assert len(ew3.c1) == 3
        tst.assert_allclose(ew3.c1, content_1[7:])
        tst.assert_allclose(ew3.c2, content_2[7:])
        tst.assert_allclose(ak.flatten(ew3.c3), ak.flatten(content_3[7:]))
        
        flat_ew = ak.flatten(ew3.c4)
        flat_content = ak.flatten(content_4[7:])
        try:
            flat_content = ak.flatten(flat_content)
            flat_ew = ak.flatten(flat_ew)
        except ValueError:  # already flat
            pass
        tst.assert_allclose(flat_ew, flat_content)

        assert np.all(["dog" in name for name in paths if name is not None])
        # give multiple lists to check
        # and check it deselects any selected_event
        ew.selected_event = 2
        paths = ew.split([0, 5, 7, 7], [5, 7, 7, 10], ["c1", "c3", "c4"] , "dog")
        ew0 = Components.EventWise.from_file(paths[0])
        assert len(ew0.c1) == 5
        tst.assert_allclose(ew0.c1, content_1[:5])
        tst.assert_allclose(ew0.c2, content_2[:5])
        tst.assert_allclose(ak.flatten(ew0.c3), ak.flatten(content_3[:5]))
        
        flat_ew = ak.flatten(ew0.c4)
        flat_content = ak.flatten(content_4[:5])
        try:
            flat_content = ak.flatten(flat_content)
            flat_ew = ak.flatten(flat_ew)
        except ValueError:  # already flat
            pass
        tst.assert_allclose(flat_ew, flat_content)

        # also check it throws an AssertionError if columns are diferent lengths
        try:
            paths = ew.split([0, 5, 7, 7], [5, 7, 7, 10],
                             ["c1", "c3", "c4"], "dog")
            raise AssertionError(
                "Should have raise error as c2 is a diferent length")
        except AssertionError:
            pass
        # check it throws an error if the lower and upper bounds don't make sense
        try:
            paths = ew.split([0, 8, 7, 7], [5, 7, 7, 10],
                             ["c1", "c3", "c4"], "dog")
            raise AssertionError(
                "Should have raised error as bounds are invalid")
        except ValueError:
            pass


def test_fragment():
    with TempTestDir("tst") as dir_name:
        file_name = "test.parquet"
        ew = Components.EventWise(os.path.join(dir_name, file_name))
        # try with 12 events
        n_events = 12
        content_1 = ak.from_iter(np.arange(n_events))
        content_2 = ak.from_iter(np.random.rand(n_events))
        content_3 = ak.from_iter([np.random.rand(np.random.randint(5)) for _ in range(n_events)])
        content_4 = ak.from_iter([[ak.from_iter(np.random.rand(np.random.randint(5)))
                                       for _ in range(np.random.randint(5))]
                                      for _ in range(n_events)])
        ew.append(c1=content_1, c2=content_2, c3=content_3, c4=content_4)
        # we can fragment 2 ways, first by number of fragments
        paths = ew.fragment('c1', n_fragments=3)
        for i, path in enumerate(paths):
            ew0 = Components.EventWise.from_file(path)
            assert len(ew0.c1) == 4
            idxs = slice(i*4, (i+1)*4)
            tst.assert_allclose(ew0.c1, content_1[idxs])
            tst.assert_allclose(ew0.c2, content_2[idxs])
            tst.assert_allclose(ak.flatten(ew0.c3), ak.flatten(content_3[idxs]))
            flat_ew = ak.flatten(ew0.c4)
            flat_content = ak.flatten(content_4[idxs])
            try:
                flat_ew = ak.flatten(flat_ew)
                flat_content = ak.flatten(flat_content)
            except ValueError:  # already flat
                pass
            tst.assert_allclose(flat_ew, flat_content)
        # we can fragment 2 ways, first by number of fragments
        paths = ew.fragment('c1', fragment_length=3)
        for i, path in enumerate(paths):
            ew0 = Components.EventWise.from_file(path)
            assert len(ew0.c1) == 3
            idxs = slice(i*3, (i+1)*3)
            tst.assert_allclose(ew0.c1, content_1[idxs])
            tst.assert_allclose(ew0.c2, content_2[idxs])
            tst.assert_allclose(ak.flatten(ew0.c3), ak.flatten(content_3[idxs]))
            flat_ew = ak.flatten(ew0.c4)
            flat_content = ak.flatten(content_4[idxs])
            try:
                flat_ew = ak.flatten(flat_ew)
                flat_content = ak.flatten(flat_content)
            except ValueError:  # already flat
                pass
            tst.assert_allclose(flat_ew, flat_content)


def test_split_unfinished():
    with TempTestDir("tst") as dir_name:
        file_name = "test.parquet"
        ew = Components.EventWise(os.path.join(dir_name, file_name))
        # try with 12 events
        n_events = 12
        n_unfinished = 2
        content_1 = ak.from_iter(np.arange(n_events))
        content_2 = ak.from_iter(np.random.rand(n_events-n_unfinished))
        content_3 = ak.from_iter([np.random.rand(np.random.randint(5)) for _ in range(n_events)])
        content_4 = ak.from_iter([[ak.from_iter(np.random.rand(np.random.randint(5)))
                                       for _ in range(np.random.randint(5)+1)]
                                      for _ in range(n_events)])
        ew.append(c1=content_1, c2=content_2, c3=content_3, c4=content_4)
        paths = ew.split_unfinished('c1', 'c2')
        ew0 = Components.EventWise.from_file(paths[0])
        assert len(ew0.c1) == n_events - n_unfinished
        idxs = slice(n_events-n_unfinished)
        tst.assert_allclose(ew0.c1, content_1[idxs])
        tst.assert_allclose(ew0.c2, content_2[idxs])
        tst.assert_allclose(ak.flatten(ew0.c3), ak.flatten(content_3[idxs]))
        tst.assert_allclose(ak.flatten(ak.flatten(ew0.c4)),
                            ak.flatten(ak.flatten(content_4[idxs])))
        ew1 = Components.EventWise.from_file(paths[1])
        assert len(ew1.c1) == n_unfinished
        idxs = slice(n_events-n_unfinished, None)
        tst.assert_allclose(ew1.c1, content_1[idxs])
        tst.assert_allclose(ak.flatten(ew1.c3), ak.flatten(content_3[idxs]))
        tst.assert_allclose(ak.flatten(ak.flatten(ew1.c4)),
                            ak.flatten(ak.flatten(content_4[idxs])))
        # try saving in another dir
        with TempTestDir(os.path.join(dir_name, "subdir")) as sub_dir:
            # also try giveing the finished and unfinished components as list
            paths = ew.split_unfinished(['c1'], ['c2'], dir_name=sub_dir)
            # should work the same from here
            ew0 = Components.EventWise.from_file(paths[0])
            assert len(ew0.c1) == n_events - n_unfinished
            idxs = slice(n_events-n_unfinished)
            tst.assert_allclose(ew0.c1, content_1[idxs])
            tst.assert_allclose(ew0.c2, content_2[idxs])
            tst.assert_allclose(ak.flatten(ew0.c3), ak.flatten(content_3[idxs]))
            tst.assert_allclose(ak.flatten(ak.flatten(ew0.c4)),
                                ak.flatten(ak.flatten(content_4[idxs])))
            ew1 = Components.EventWise.from_file(paths[1])
            assert len(ew1.c1) == n_unfinished
            idxs = slice(n_events-n_unfinished, None)
            tst.assert_allclose(ew1.c1, content_1[idxs])
            tst.assert_allclose(ak.flatten(ew1.c3), ak.flatten(content_3[idxs]))
            tst.assert_allclose(ak.flatten(ak.flatten(ew1.c4)),
                                ak.flatten(ak.flatten(content_4[idxs])))
        # try with all events unfinished
        ew.append(c2=ak.from_iter([]))
        paths = ew.split_unfinished('c1', 'c2')
        assert paths[0] is None, f"Expected the first path to be None, instead paths are {paths}"
        assert isinstance(paths[1], str), f"Expected the second path to be a string, instead paths are {paths}"

        # try with no unfinished events
        ew.append(c2=content_1)
        paths = ew.split_unfinished('c1', 'c2')
        assert paths[1] is None, f"Expected the second path to be None, instead paths are {paths}"
        assert isinstance(paths[0], str), f"Expected the first path to be a string, instead paths are {paths}"


def test_combine():
    with TempTestDir("tst") as dir_name:
        # splitting a blank ew should result in only Nones
        file_name = "test.parquet"
        ew = Components.EventWise(os.path.join(dir_name, file_name))
        # try with 10 events
        n_events = 10
        content_1 = ak.from_iter(np.arange(n_events))
        content_2 = ak.from_iter(np.random.rand(n_events))
        content_3 = ak.from_iter([np.random.rand(np.random.randint(5)) for _ in range(n_events)])
        content_4 = ak.from_iter([[ak.from_iter(np.random.rand(np.random.randint(5)))
                                   for _ in range(np.random.randint(5))]
                                  for _ in range(n_events)])
        hyper = AwkdArrays.one_one
        ew.append(Event_n=content_1, c2=content_2, c3=content_3, c4=content_4)
        ew.append_hyperparameters(Hyper=hyper)
        paths = ew.split([0, 5, 7, 7], [5, 7, 7, 10], "Event_n", "dog")
        # delete the original
        os.remove(os.path.join(dir_name, file_name))
        subdir_name = os.path.split(paths[0])[0]
        # combine the fragments
        recombined = Components.EventWise.combine(subdir_name, "test", del_fragments=False, check_for_dups=False)
        # tere in no order garentee, so get the new order from Event_n
        order = np.argsort(recombined.Event_n)
        tst.assert_allclose(recombined.Event_n[order], content_1)
        tst.assert_allclose(recombined.c2[order], content_2)
        tst.assert_allclose(ak.flatten(recombined.c3[order]), ak.flatten(content_3))
        for i in range(n_events):
            tst.assert_allclose(ak.flatten(recombined.c4[order[i]]),
                                ak.flatten(content_4[i]))
        assert generic_equality_comp(recombined.Hyper, AwkdArrays.one_one)
        # delete the combination
        os.remove(recombined.path_name)
        # check if check_for_dups prevents adding the same column multiple times
        dup = ak.from_iter([7, 8])
        for path in paths:
            if path is None:
                continue
            ew = Components.EventWise.from_file(path)
            ew.append(Dup=dup)
        # combine the fragments
        recombined = Components.EventWise.combine(subdir_name, "test", check_for_dups=True, del_fragments=True)
        assert len(recombined.Dup) == 2
        # check for dups should not otehrwise change the content
        order = np.argsort(recombined.Event_n)
        tst.assert_allclose(recombined.Event_n[order], content_1)
        tst.assert_allclose(recombined.c2[order], content_2)
        tst.assert_allclose(ak.flatten(recombined.c3[order]), ak.flatten(content_3))
        for i in range(n_events):
            tst.assert_allclose(ak.flatten(recombined.c4[order[i]]),
                                ak.flatten(content_4[i]))
        assert generic_equality_comp(recombined.Hyper, AwkdArrays.one_one)
        # check if the fragments were deleted
        assert len(os.listdir(dir_name)) == 1


def test_recursive_combine():
    with TempTestDir("tst") as dir_name:
        dir_name += '/'
        file_name = "test.parquet"
        ew = Components.EventWise(os.path.join(dir_name, file_name))
        # try with 10 events
        n_events = 10
        content_1 = ak.from_iter(np.arange(n_events))
        content_2 = ak.from_iter(np.random.rand(n_events))
        ew.append(c1=content_1, c2=content_2)
        paths = ew.split([0, 5, 7, 7], [5, 7, 7, 10], "c1", "dog")
        del ew
        os.remove(os.path.join(dir_name, file_name))  # so it isn't added twice
        resplit = Components.EventWise.from_file(paths[0])
        resplit.fragment("c1", n_fragments=3)
        os.remove(paths[0])  # so it isn't added twice
        dir_name = os.path.split(paths[0])[0]
        recombined = Components.EventWise.recursive_combine(dir_name)
        recombined = Components.EventWise.from_file(recombined)
        # tere in no order garentee, so get the new order from c1
        order = np.argsort(recombined.c1)
        tst.assert_allclose(recombined.c1[order], content_1)
        tst.assert_allclose(recombined.c2[order], content_2)


# out of eventwise ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_event_matcher():
    with TempTestDir("tst") as dir_name:
        # splitting a blank ew should result in only Nones
        file_nameA = "testA.parquet"
        ewA = Components.EventWise(os.path.join(dir_name, file_nameA))
        # make a longer eventwise
        n_events = 10
        #                              0  1  2  3  4  5  6  7  8  9
        content_A1 = ak.from_iter([0, 1, 2, 3, 3, 3, 4, 5, 6, 6])
        content_A2 = ak.from_iter(0.5*np.arange(n_events*2).reshape(n_events, 2))
        content_A3 = ak.from_iter([0, 1, 2, 3, 3, 9, 4, 5, 6, 6])
        hyper = AwkdArrays.one_one
        ewA.append(Event_n=content_A1, c2=content_A2, c3=content_A3)
        ewA.append_hyperparameters(Hyper=hyper)
        # make an eventWise with a subset of the longer ones events
        file_nameB = "testB.parquet"
        ewB = Components.EventWise(os.path.join(dir_name, file_nameB))
        required_order = [5, 3, 2, 8, 4]
        content_B1 = content_A1[required_order]
        content_B2 = content_A2[required_order]
        content_B3 = content_A3[required_order]
        hyper = AwkdArrays.one_one
        ewB.append(Event_n=content_B1, c2=content_B2, c3=content_B3)
        ewB.append_hyperparameters(Hyper=hyper)
        # the arrangment required to make B into A
        expected = np.array([-1, -1, 2, 1, 4, 0, -1, -1, 3, -1])
        found = Components.event_matcher(ewA, ewB)
        assert np.all(expected == found), f"expected = {expected}, found={found}"


def test_recursive_distance():
    # for arrays of diferign shape the distance should be infinite
    found = Components.recursive_distance(AwkdArrays.jet_ints, AwkdArrays.jet_ints[:1])
    assert found == np.Inf
    found = Components.recursive_distance(AwkdArrays.jet_ints, AwkdArrays.one_one)
    assert found == np.Inf
    version1 = ak.from_iter([0,[1,2]])
    version2 = ak.from_iter([-1,[1,2]])
    found = Components.recursive_distance(version1, version1)
    assert found == 0
    found = Components.recursive_distance(version2, version1)
    assert found == 1


def test_add_rapidity():
    large_num = np.inf
    pts = ak.from_iter([[0., 0., 0., 0.,  0., 1., 1.,  1., 1., 10.]])
    pzs = ak.from_iter([[0., 0., 1., -1., 1., 1., -1., 0., 0., 10.]])
    es =  ak.from_iter([[0., 1., 1., 1.,  2., 2., 2.,  1., 2., 100.]])
    rap = ak.from_iter([np.nan, 0., large_num + 1., -large_num-1., 0.5*np.log(3/1),
                            0.5*np.log(3/1), -0.5*np.log(3/1), 0., 0., 0.5*np.log(110./90.)])
    with TempTestDir("tst") as dir_name:
        # instansation
        file_name = "rapidity.parquet"
        contents = {"PT": pts, "Pz": pzs, "Energy": es}
        ew = Components.EventWise(os.path.join(dir_name, file_name),
                                  columns=list(contents.keys()),
                                  contents=contents)
        Components.add_rapidity(ew)
        ew.selected_event = 0
        tst.assert_allclose(ew.Rapidity, rap)
        # try adding to a specific prefix
        contents = {"A_PT": pts, "A_Pz": pzs, "A_Energy": es,
                    "B_PT": pts, "B_Pz": pzs, "B_Energy": es}
        ew = Components.EventWise(os.path.join(dir_name, file_name),
                                  columns=list(contents.keys()),
                                  contents=contents)
        Components.add_rapidity(ew, 'A')
        ew.selected_event = 0
        tst.assert_allclose(ew.A_Rapidity, rap)
        with pytest.raises(AttributeError):
            ew.B_Rapidity


def test_add_mass():
    large_num = np.inf
    pxs = ak.from_iter([[0., 1., 0., 0.,  0., 1., 1.,  1., 1., 10.]])
    pys = ak.from_iter([[0., 0., 1., 1., 0., 1., -1., 0., 0., 10.]])
    pzs = ak.from_iter([[0., 0., 1., 0., 0., 1., -1., 0., 0., 10.]])
    es =  ak.from_iter([[0., 1., 2., 3.,  2., 5., 4.,  1., 2., 100.]])
    mass = ak.from_iter([0, 0., np.sqrt(2), np.sqrt(9-1), 2.,
                            np.sqrt(25-3), np.sqrt(16-3), 0.,
                            np.sqrt(4-1), np.sqrt(10000 - 300)])
    with TempTestDir("tst") as dir_name:
        # instansation
        file_name = "rapidity.parquet"
        contents = {"Px": pxs, "Py": pys, "Pz": pzs, "Energy": es}
        ew = Components.EventWise(os.path.join(dir_name, file_name),
                                  columns=list(contents.keys()),
                                  contents=contents)
        Components.add_mass(ew)
        ew.selected_event = 0
        tst.assert_allclose(ew.Mass, mass)
        # try adding to a specific prefix
        contents = {"A_Px": pxs, "A_Py": pys, "A_Pz": pzs, "A_Energy": es,
                    "B_Px": pxs, "B_Py": pys, "B_Pz": pzs, "B_Energy": es}
        ew = Components.EventWise(os.path.join(dir_name, file_name),
                                  columns=list(contents.keys()),
                                  contents=contents)
        Components.add_mass(ew, 'A')
        ew.selected_event = 0
        tst.assert_allclose(ew.A_Mass, mass)
        with pytest.raises(AttributeError):
            ew.B_Mass


class Particle:
    def __init__(self, direction, mass):
        self.px = ak.from_iter([0.])
        self.py = ak.from_iter([0.])
        self.pz = ak.from_iter([0.])
        self.pt = ak.from_iter([1.])
        self.p = ak.from_iter([1.])
        if direction == 'x':
            self.px = ak.from_iter([1.])
        elif direction == '-x':
            self.px = ak.from_iter([-1.])
        elif direction == 'y':
            self.py = ak.from_iter([1.])
        elif direction == '-y':
            self.py = ak.from_iter([-1.])
        elif direction == 'z':
            self.pz = ak.from_iter([1.])
            self.pt = ak.from_iter([0.])
        elif direction == '-z':
            self.pz = ak.from_iter([-1.])
            self.pt = ak.from_iter([0.])
        elif direction == '45':
            self.px = ak.from_iter([np.sqrt(0.5)])
            self.py = ak.from_iter([np.sqrt(0.5)])
            self.pz = ak.from_iter([1.])
            self.p = ak.from_iter([np.sqrt(2.)])
        self.m2 = ak.from_iter([mass**2])
        self.e2 = ak.from_iter([self.m2[0] + self.p[0]**2])
        self.e = np.sqrt(self.e2)
        self.et = self.e * (self.pt/self.p)


def test_add_thetas():
    # particles could go down each axial direction
    input_output = [
            ('x', np.pi/2.),
            ('-x', np.pi/2.),
            ('y', np.pi/2.),
            ('-y', np.pi/2.),
            ('z', 0.),
            ('-z', np.pi),
            ('45', np.pi/4.)]
    with TempTestDir("tst") as dir_name:
        # instansation
        file_name = "rapidity.parquet"
        for inp, out in input_output:
            particle = Particle(inp, 0)
            # there are many things theta can be calculated from
            # pz&birr pt&pz (px&py)&pz pt&birr et&e
            contents = {"Birr": particle.p, "Pz": particle.pz}
            ew = Components.EventWise(os.path.join(dir_name, file_name),
                                      columns=list(contents.keys()), 
                                      contents=contents)
            Components.add_thetas(ew, '')
            tst.assert_allclose(ew.Theta[0], out)
            contents = {"PT": particle.pt, "Pz": particle.pz}
            ew = Components.EventWise(os.path.join(dir_name, file_name),
                                      columns=list(contents.keys()),
                                      contents=contents)
            Components.add_thetas(ew, '')
            tst.assert_allclose(ew.Theta[0], out)
            contents = {"Px": particle.px, "Py": particle.py, "Pz": particle.pz}
            ew = Components.EventWise(os.path.join(dir_name, file_name),
                                      columns=list(contents.keys()),
                                      contents=contents)
            Components.add_thetas(ew, '')
            tst.assert_allclose(ew.Theta[0], out)
            contents = {"Birr": particle.p, "Pz": particle.pz}
            ew = Components.EventWise(os.path.join(dir_name, file_name),
                                      columns=list(contents.keys()),
                                      contents=contents)
            Components.add_thetas(ew, '')
            tst.assert_allclose(ew.Theta[0], out)
            # Need to add energy version for towers, trouble getting direction
            #contents = {"Energy": particle.e, "ET": particle.et}
            #ew = Components.EventWise(os.path.join(dir_name, file_name),
            #                          columns=list(contents.keys()),
            #                          contents=contents)
            #Components.add_thetas(ew, '')
            #tst.assert_allclose(ew.Theta[0], out)
            # check that adding mass makes no diference
            particle = Particle(inp, 1.)
            contents = {"Birr": particle.p, "Pz": particle.pz}
            ew = Components.EventWise(os.path.join(dir_name, file_name),
                                      columns=list(contents.keys()),
                                      contents=contents)
            Components.add_thetas(ew, '')
            tst.assert_allclose(ew.Theta[0], out)
            contents = {"PT": particle.pt, "Pz": particle.pz}
            ew = Components.EventWise(os.path.join(dir_name, file_name),
                                      columns=list(contents.keys()),
                                      contents=contents)
            Components.add_thetas(ew, '')
            tst.assert_allclose(ew.Theta[0], out)
            contents = {"Px": particle.px, "Py": particle.py, "Pz": particle.pz}
            ew = Components.EventWise(os.path.join(dir_name, file_name),
                                      columns=list(contents.keys()),
                                      contents=contents)
            Components.add_thetas(ew, '')
            tst.assert_allclose(ew.Theta[0], out)
            contents = {"Birr": particle.p, "Pz": particle.pz}
            ew = Components.EventWise(os.path.join(dir_name, file_name),
                                      columns=list(contents.keys()),
                                      contents=contents)
            Components.add_thetas(ew, '')
            tst.assert_allclose(ew.Theta[0], out)
            # Need to add energy version for towers, trouble getting direction
            #contents = {"Energy": particle.e, "ET": particle.et}
            #ew = Components.EventWise(os.path.join(dir_name, file_name),
            #                          columns=list(contents.keys()),
            #                          contents=contents)
            #Components.add_thetas(ew, '')
            #tst.assert_allclose(ew.Theta[0], out)
            # try letting the function find the naems
            contents = {"Dog_Birr": particle.p, "Dog_Pz": particle.pz, "Dog_Phi": particle.pz}
            ew = Components.EventWise(os.path.join(dir_name, file_name),
                                      columns=list(contents.keys()),
                                      contents=contents)
            Components.add_thetas(ew, None)
            tst.assert_allclose(ew.Dog_Theta[0], out)
            # try letting the fix the prefix
            contents = {"Dog_Birr": particle.p, "Dog_Pz": particle.pz}
            ew = Components.EventWise(os.path.join(dir_name, file_name),
                                      columns=list(contents.keys()),
                                      contents=contents)
            Components.add_thetas(ew, "Dog")
            tst.assert_allclose(ew.Dog_Theta[0], out)
            # give it an impossible task
            contents = {"Dog_Birr": particle.p, "Dog_Phi": particle.pz}
            ew = Components.EventWise(os.path.join(dir_name, file_name),
                                      columns=list(contents.keys()),
                                      contents=contents)
            Components.add_thetas(ew, "Dog")
            assert "Dog_Theta" not in ew.columns


def test_theta_to_pseudorapidity():
    input_output = [
            (0., np.inf),
            (np.pi/2, 0.),
            (3, -np.log(np.tan(1.5))),
            (np.pi, -np.inf)]
    for inp, out in input_output:
        etas = Components.theta_to_pseudorapidity(np.array([inp]))
        tst.assert_allclose(etas[0], out, atol=0.0001)
    # try as a float
    eta = Components.theta_to_pseudorapidity(input_output[0][0])
    assert isinstance(eta, float), f"eta={eta} and has type {type(eta)}"


def test_add_pseudorapidity():
    large_num = np.inf
    theta = ak.from_iter([[0., np.pi/4, np.pi/2, 3*np.pi/4, np.pi]])
    eta = ak.from_iter([np.inf, -np.log(np.tan(np.pi/8)), 0., np.log(np.tan(np.pi/8)), -np.inf])
    with TempTestDir("tst") as dir_name:
        # instansation
        file_name = "pseudorapidity.parquet"
        contents = {"Theta": theta}
        ew = Components.EventWise(os.path.join(dir_name, file_name),
                                  columns=list(contents.keys()),
                                  contents=contents)
        Components.add_pseudorapidity(ew)
        ew.selected_event = 0
        tst.assert_allclose(ew.PseudoRapidity, eta, atol=0.0001)
        # try adding to a specific prefix
        contents = {"A_Theta": theta,
                    "B_Theta": theta}
        ew = Components.EventWise(os.path.join(dir_name, file_name),
                                  columns=list(contents.keys()),
                                  contents=contents)
        Components.add_pseudorapidity(ew, 'A')
        ew.selected_event = 0
        tst.assert_allclose(ew.A_PseudoRapidity, eta, atol=0.0001)
        with pytest.raises(AttributeError):
            ew.B_PseudoRapidity


def test_add_phi():
    input_output = [
            ({"Px": 1, "Py": 0, "Pz": 10}, 0.),
            ({"Px": 1, "Py": 0, "Pz": 0}, 0.),
            ({"Px": -1, "Py": 0, "Pz": 0}, np.pi),
            ({"Px": 0, "Py": 1, "Pz": 0}, np.pi/2),
            ({"Px": 0, "Py": -1, "Pz": 0}, -np.pi/2),]
    with TempTestDir("tst") as dir_name:
        # instansation
        file_name = "phi.parquet"
        for contents, out in input_output:
            contents = {key: ak.from_iter([[value]]) for key, value
                        in contents.items()}
            ew = Components.EventWise(os.path.join(dir_name, file_name),
                                      columns=list(contents.keys()),
                                      contents=contents)
            Components.add_phi(ew, '')
            tst.assert_allclose(ew.Phi[0], out)
        # try letting the function idetify valid columns
        contents = {"Px": ak.from_iter([[0]]), "Py": ak.from_iter([[1]]),
                    "Dog_Px": ak.from_iter([[1]]), "Dog_Py": ak.from_iter([[0]])}
        ew = Components.EventWise(os.path.join(dir_name, file_name),
                                  columns=list(contents.keys()),
                                  contents=contents)
        Components.add_phi(ew)
        tst.assert_allclose(ew.Phi[0], np.pi/2)
        tst.assert_allclose(ew.Dog_Phi[0], 0)
        # let the function fix a basename
        contents = {"Dog_Px": ak.from_iter([[1]]), "Dog_Py": ak.from_iter([[1]])}
        ew = Components.EventWise(os.path.join(dir_name, file_name),
                                  columns=list(contents.keys()),
                                  contents=contents)
        Components.add_phi(ew, "Dog")
        tst.assert_allclose(ew.Dog_Phi[0], np.pi/4)


def test_add_PT():
    # particles could go down each axial direction
    input_output = [
            ('x', 1.),
            ('-x', 1.),
            ('y', 1.),
            ('-y', 1.),
            ('z', 0.),
            ('-z', 0.),
            ('45', 1.)]
    with TempTestDir("tst") as dir_name:
        # instansation
        file_name = "PT.parquet"
        for inp, out in input_output:
            particle = Particle(inp, 0)
            contents = {"Px": particle.px, "Py": particle.py}
            ew = Components.EventWise(os.path.join(dir_name, file_name),
                                      columns=list(contents.keys()), 
                                      contents=contents)
            Components.add_PT(ew, '')
            tst.assert_allclose(ew.PT[0], out)
        # try letting the function idetify valid columns
        contents = {"Px": particle.px, "Py": particle.py,
                    "Dog_Px": particle.px, "Dog_Py": particle.py}
        ew = Components.EventWise(os.path.join(dir_name, file_name),
                                  columns=list(contents.keys()), 
                                  contents=contents)
        Components.add_PT(ew)
        tst.assert_allclose(ew.PT[0], out)
        tst.assert_allclose(ew.Dog_PT[0], out)
        # let the function fix a basename
        contents = {"Dog_Px": particle.px, "Dog_Py": particle.py}
        ew = Components.EventWise(os.path.join(dir_name, file_name),
                                  columns=list(contents.keys()), 
                                  contents=contents)
        Components.add_PT(ew, "Dog")
        tst.assert_allclose(ew.Dog_PT[0], out)


def test_RootReadout():
    root_file = os.path.join(data_dir, "mini.root")
    components = ["Particle", "Track", "Tower"]
    rr = Components.RootReadout(root_file, components)
    n_events = len(rr.Energy)
    test_events = np.random.randint(0, n_events, 20)
    idents = PDGNames.Identities()
    all_ids = set(idents.particle_data[:, idents.columns["id"]])
    for event_n in test_events:
        rr.selected_event = event_n
        # sanity checks on particle values
        # momentum
        tst.assert_allclose(rr.PT**2, (rr.Px**2 + rr.Py**2), atol=0.001, rtol=0.01)
        tst.assert_allclose(rr.Birr**2, (rr.PT**2 + rr.Pz**2), atol=0.001, rtol=0.01)
        # on shell  lol-it's not on shell
        #tst.assert_allclose(rr.Mass, (rr.Energy**2 - rr.Birr**2), atol=0.0001)
        # angles
        tst.assert_allclose(rr.Phi, np.arctan2(rr.Py, rr.Px), atol=0.001, rtol=0.01)
        m2 = rr.Energy**2 - rr.Pz**2 - rr.PT**2
        with np.errstate(divide='ignore', invalid='ignore'):
            rapidity_calculated = 0.5*np.log((rr.PT**2 + m2)/(rr.Energy - np.abs(rr.Pz))**2)
        rapidity_calculated = ak.to_numpy(rapidity_calculated)
        rapidity_calculated[np.isnan(rapidity_calculated)] = np.inf
        rapidity_calculated *= np.sign(ak.to_numpy(rr.Pz))
        # im having dificulty matching the rapitity cauclation at all infinite points
        # this probably dosn't matter as high rapidity values are not seen anyway
        with np.errstate(invalid='ignore'):
            filt = np.logical_and(np.abs(rr.Rapidity) < 7., np.isfinite(rapidity_calculated))
        tst.assert_allclose(rr.Rapidity[filt], rapidity_calculated[filt], atol=0.01, rtol=0.01)
        # valid PID
        assert set(np.abs(rr.PID)).issubset(all_ids)
        # sanity checks on Towers
        # energy
        assert np.all(rr.Tower_ET <= rr.Tower_Energy)
        tst.assert_allclose((rr.Tower_Eem + rr.Tower_Ehad), rr.Tower_Energy, atol=0.001)
        # angle
        theta = np.arcsin(rr.Tower_ET/rr.Tower_Energy)
        eta = - np.log(np.tan(theta/2))
        tst.assert_allclose(eta, np.abs(rr.Tower_Eta), atol=0.001)
        # particles
        assert np.all(ak.flatten(rr.Tower_Particles) >= 0)
        num_hits = [len(p) for p in rr.Tower_Particles]
        tst.assert_allclose(num_hits, rr.Tower_NTimeHits)
        # sanity check on Tracks
        # momentum
        # mometum removes as it appears to old nonsense
        #assert np.all(rr.Track_PT <= rr.Track_Birr)
        # angle
        #theta = np.arcsin(rr.Track_PT/rr.Track_Birr)
        #eta = - np.log(np.tan(theta/2))
        #tst.assert_allclose(eta, np.abs(rr.Track_Eta), atol=0.001)
        # particles
        assert np.all(rr.Track_Particle >= 0)
    

def test_edge_instance():
    # find the last particle
    input_output = [
            ([0], {"MCPID": [1], "Children": [[]]}, {0}),  # lone particle return itself
            ([0], {"MCPID": [1, 2], "Children": [[1], []]}, {0}),  # particle decays immediatly
            ([0], {"MCPID": [1, 1, 2], "Children": [[1], [2], []]}, {1}),  # particle decays after 1 step
            ([0, 1], {"MCPID": [1, 1, 2], "Children": [[1], [2], []]}, {1}),  # particle decays after 1 step, two particles given
            ([1], {"MCPID": [2, 1, 1], "Children": [[], [2], [0]]}, {2}),  # particle decays out of order
            ([2], {"MCPID": [2, 1, 1], "Children": [[], [2], [0]]}, {2}),  # particle decays out of order
            ([0], {"MCPID": [1, 2, 1], "Children": [[1, 2], [], []]}, {2}),  # multiple children
            ([0, 2], {"MCPID": [1, 2, 1], "Children": [[1, 2], [], []]}, {2}),  # multiple children, two particles given
            ([0], {"MCPID": [1, 1, 1], "Children": [[1, 2], [], []]}, {1, 2}),  # multiple ends
            ([1, 2], {"MCPID": [2, 1, 1], "Children": [[1, 2], [], []]}, {1, 2}),  # multiple ends, two particles given
            ([1, 2], {"MCPID": [2, 1, 1, 1], "Children": [[1, 2], [], [3], []]}, {1, 3})]  # multiple ends, two particles given
    with TempTestDir("tst") as dir_name:
        # instansation
        file_name = "last.parquet"
        for start, contents, expected in input_output:
            contents = {key: ak.from_iter([value]) for key, value
                        in contents.items()}
            ew = Components.EventWise(os.path.join(dir_name, file_name),
                                      columns=list(contents.keys()),
                                      contents=contents)
            ew.selected_event = 0
            found = Components.edge_instance(ew, *start, first=False)
            assert found == expected, f"For setup {contents}, expected {expected} found {found}"
            # flip to find first
            contents["Parents"] = contents["Children"]
            del contents["Children"]
            ew = Components.EventWise(os.path.join(dir_name, file_name),
                                      columns=list(contents.keys()),
                                      contents=contents)
            ew.selected_event = 0
            found = Components.edge_instance(ew, *start, first=True)
            assert found == expected, f"For setup {contents}, expected {expected} found {found}"
        # calling on an empty set should return an empty set
        found = Components.edge_instance(ew)
        assert len(found) == 0



def test_fix_nonexistent_columns():
    with TempTestDir("tst") as dir_name:
        file_name = "blank.parquet"
        blank_ew = Components.EventWise(os.path.join(dir_name, file_name))
        # should have in effect on a blank eventWise
        h_problems, blank_ew = Components.fix_nonexistent_columns(blank_ew)
        assert len(h_problems) == 0
        assert len(blank_ew.columns) == 0
        assert len(blank_ew.hyperparameter_columns) == 0
        # should not remove anything from a valid eventWise
        contents = {"A": AwkdArrays.one_one, "B": AwkdArrays.empty}
        blank_ew.append(**contents)
        h_problems, blank_ew = Components.fix_nonexistent_columns(blank_ew)
        assert len(h_problems) == 0
        assert generic_equality_comp(blank_ew.A, contents["A"])
        assert generic_equality_comp(blank_ew.B, contents["B"])
        # should remove eronious objects
        blank_ew.columns += ["oops"]
        blank_ew.hyperparameter_columns += ["oops2"]
        h_problems, blank_ew = Components.fix_nonexistent_columns(blank_ew)
        assert len(h_problems) == 1
        assert "oops2" == h_problems[0]
        assert len(blank_ew.columns) == 2
        assert generic_equality_comp(blank_ew.A, contents["A"])
        assert generic_equality_comp(blank_ew.B, contents["B"])


def test_even_length():
    with TempTestDir("tst") as dir_name:
        file_name = "blank.parquet"
        blank_ew = Components.EventWise(os.path.join(dir_name, file_name))
        # should have no effect on a blank eventWise
        Components.check_even_length(blank_ew, True)
        assert len(blank_ew.columns) == 0
        assert len(blank_ew.hyperparameter_columns) == 0
        # should not remove anything from a valid eventWise
        contents = {"Thing_A": AwkdArrays.jet_ints, "Thing_B": AwkdArrays.jet_floats}
        blank_ew.append(**contents)
        Components.check_even_length(blank_ew, True)
        assert generic_equality_comp(blank_ew.Thing_A, contents["Thing_A"])
        assert generic_equality_comp(blank_ew.Thing_B, contents["Thing_B"])
        # should throw errors with one column the wrong length
        wrong_content = {"Wrong_A": AwkdArrays.jet_ints[:1], "Wrong_B": AwkdArrays.jet_floats}
        blank_ew.append(**wrong_content)
        with pytest.raises(ValueError):
            Components.check_even_length(blank_ew, True)
        # should not raise an error if told to ignore the prefix
        Components.check_even_length(blank_ew, True, ["Wrong"])
        # should remove the prefix if not in interactive mode and not throwing errors
        Components.check_even_length(blank_ew, False)
        assert len(blank_ew.columns) == 2, f"Expected [Thing_A, Thing_B], found {blank_ew.columns}"
        assert generic_equality_comp(blank_ew.Thing_A, contents["Thing_A"])
        assert generic_equality_comp(blank_ew.Thing_B, contents["Thing_B"])

def test_check_no_tachions():
    energies = ak.from_iter([[1., 0., 10.]])
    px = ak.from_iter([[0.5, 0., 7.]])
    py = ak.from_iter([[-0.5, 0., np.sqrt(100. - 7**2)]])
    pz = ak.from_iter([[0., 0., 0.]])
    bad_pz = ak.from_iter([[-1, 0, 1]])
    with TempTestDir("tst") as dir_name:
        file_name = "blank.parquet"
        blank_ew = Components.EventWise(os.path.join(dir_name, file_name))
        # should have no effect on a blank eventWise
        Components.check_no_tachions(blank_ew, True)
        assert len(blank_ew.columns) == 0
        assert len(blank_ew.hyperparameter_columns) == 0
        # should not remove anything from a valid eventWise
        contents = {"Thing_Energy": energies, "Thing_Px": px,
                    "Thing_Py": py, "Thing_Pz": pz}
        blank_ew.append(**contents)
        Components.check_no_tachions(blank_ew, True)
        assert generic_equality_comp(blank_ew.Thing_Energy, energies)
        assert generic_equality_comp(blank_ew.Thing_Px, px)
        assert generic_equality_comp(blank_ew.Thing_Py, py)
        assert generic_equality_comp(blank_ew.Thing_Pz, pz)
        # should throw errors with one column the wrong length
        blank_ew.append(Thing_Pz=bad_pz)
        with pytest.raises(ValueError):
            Components.check_no_tachions(blank_ew, True)
        # should not raise an error if told to ignore the prefix
        Components.check_no_tachions(blank_ew, True, ["Thing"])
        # should remove the prefix if not in interactive mode and not throwing errors
        Components.check_no_tachions(blank_ew, False)
        assert len(blank_ew.columns) == 0


def test_find_eventWise_in_dir():
    with TempTestDir("tst") as dir_name:
        # splitting a blank ew should result in only Nones
        file_name = "test.parquet"
        ew = Components.EventWise(os.path.join(dir_name, file_name))
        # try with 10 events
        n_events = 10
        content_1 = ak.from_iter(np.arange(n_events))
        content_2 = ak.from_iter(np.random.rand(n_events))
        content_3 = ak.from_iter([np.random.rand(np.random.randint(5)) for _ in range(n_events)])
        content_4 = ak.from_iter([[ak.from_iter(np.random.rand(np.random.randint(5)))
                                       for _ in range(np.random.randint(5))]
                                      for _ in range(n_events)])
        ew.append(c1=content_1, c2=content_2, c3=content_3, c4=content_4)
        paths = ew.split([0, 5, 7, 7], [5, 7, 7, 10], "c1", "dog")
        paths = [name for name in paths if name is not None]
        subdir_name = os.path.join(dir_name, "test_dog")
        found = Components.find_eventWise_in_dir(subdir_name)
        assert len(found) == len(paths), f"expected {paths}, found {found}"
        for ew in found:
            assert ew.path_name in paths


def test_typify():
    # check that empty arrays are given a type and don't change shape
    empty0 = ak.Array([])
    typed0 = Components.typify(empty0)
    assert 'unknown' not in str(typed0.type)
    assert len(typed0) == 0
    empty1 = ak.Array([[]])
    typed1 = Components.typify(empty1)
    assert 'unknown' not in str(typed1.type)
    assert len(typed1) == 1
    assert len(typed1[0]) == 0
    empty2 = ak.Array([[[]], []])
    typed2 = Components.typify(empty2)
    assert 'unknown' not in str(typed2.type)
    assert len(typed2) == 2
    assert len(typed2[0]) == 1
    assert len(typed2[0][0]) == 0
    assert len(typed2[1]) == 0
    # check that an array that has a type is not altered
    full3 = ak.Array([6])
    typed3 = Components.typify(full3)
    assert full3 == typed3
    assert full3.type == typed3.type
    full4 = ak.Array([6.5])
    typed4 = Components.typify(full4)
    assert full4 == typed4
    assert full4.type == typed4.type



