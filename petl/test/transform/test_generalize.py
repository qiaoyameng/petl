# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


from itertools import product

import pytest

from petl.errors import GeneralizationError
from petl.test.helpers import ieq
from petl.transform.generalize import generalize


def _hierarchies():
    return {
        'age': {
            'levels': [
                {'loss': 0, 'mapping': {
                    25: 25, 27: 27, 41: 41, 45: 45, 62: 62, 68: 68}},
                {'loss': 1, 'mapping': {
                    25: 'under40', 27: 'under40',
                    41: '40to60', 45: '40to60',
                    62: 'over60', 68: 'over60'}},
                {'loss': 3, 'mapping': {
                    'under40': 'any', '40to60': 'any', 'over60': 'any'}},
            ],
        },
        'city': {
            'levels': [
                {'loss': 0, 'mapping': {
                    'Beijing': 'Beijing', 'Shanghai': 'Shanghai',
                    'Shenzhen': 'Shenzhen'}},
                {'loss': 1, 'mapping': {
                    'Beijing': 'north', 'Shanghai': 'south',
                    'Shenzhen': 'south'}},
                {'loss': 2, 'mapping': {'north': 'any', 'south': 'any'}},
            ],
        },
        'gender': {
            'levels': [
                {'loss': 0, 'mapping': {'F': 'F', 'M': 'M'}},
                {'loss': 5, 'mapping': {'F': 'any', 'M': 'any'}},
            ],
        },
    }


def _table():
    return [
        ['id', 'age', 'city', 'gender', 'note'],
        [1, 25, 'Beijing', 'F', 'a'],
        [2, 27, 'Beijing', 'F', 'b'],
        [3, 41, 'Shanghai', 'M', 'c'],
        [4, 45, 'Shenzhen', 'M', 'd'],
        [5, 62, 'Shenzhen', 'M', 'e'],
        [6, 68, 'Shenzhen', 'F', 'f'],
        [6, 68, 'Shenzhen', 'F', 'f'],
    ]


def _feasible(table, hierarchies, combo, k):
    label_tables = []
    for field, levels in ((f, hierarchies[f]['levels'])
                          for f in hierarchies):
        per_depth = [levels[0]['mapping']]
        current = levels[0]['mapping']
        for level in levels[1:]:
            upper = level['mapping']
            composed = {raw: upper[label] for raw, label in current.items()}
            per_depth.append(composed)
            current = composed
        label_tables.append(per_depth)
    fields = list(hierarchies)
    header = table[0]
    counts = {}
    for row in table[1:]:
        key = tuple(label_tables[pos][combo[pos]][row[header.index(f)]]
                    for pos, f in enumerate(fields))
        counts[key] = counts.get(key, 0) + 1
    return all(count >= k for count in counts.values())


def test_generalize_matches_full_enumeration():
    table = _table()
    hierarchies = _hierarchies()
    view = generalize(table, hierarchies, k=2)

    widths = [len(spec['levels']) for spec in hierarchies.values()]
    expected_combo = None
    expected_loss = None
    for combo in product(*(range(width) for width in widths)):
        if not _feasible(table, hierarchies, combo, 2):
            continue
        loss = sum(spec['levels'][depth]['loss']
                   for spec, depth in zip(hierarchies.values(), combo))
        if expected_loss is None or loss < expected_loss:
            expected_loss = loss
            expected_combo = combo

    assert list(view.selected_levels.values()) == list(expected_combo)
    assert view.total_loss == expected_loss


def test_three_columns_three_levels_enumerate_all_27_combinations():
    table = [['c1', 'c2', 'c3']]
    raw1 = ['a', 'b', 'c']
    raw2 = ['p', 'q', 'r']
    raw3 = ['x', 'y', 'z']
    for value1 in raw1:
        for value2 in raw2:
            for value3 in raw3:
                table.append([value1, value2, value3])

    def hierarchy(raw, mid_prefix, loss_top):
        mid = {value: mid_prefix + value for value in raw}
        return {'levels': [
            {'loss': 0, 'mapping': {value: value for value in raw}},
            {'loss': 1, 'mapping': mid},
            {'loss': loss_top,
             'mapping': {label: 'all' for label in mid.values()}},
        ]}

    hierarchies = {
        'c1': hierarchy(raw1, 'm', 3),
        'c2': hierarchy(raw2, 'n', 4),
        'c3': hierarchy(raw3, 'o', 5),
    }

    def reference(k):
        best_combo = None
        best_loss = None
        feasible = []
        for combo in product(range(3), range(3), range(3)):
            mappings = []
            for pos, depth in enumerate(combo):
                field = ['c1', 'c2', 'c3'][pos]
                levels = hierarchies[field]['levels']
                composed = levels[0]['mapping']
                for upper in range(1, depth + 1):
                    upper_mapping = levels[upper]['mapping']
                    composed = {raw: upper_mapping[label]
                                for raw, label in composed.items()}
                mappings.append(composed)
            counts = {}
            for row in table[1:]:
                key = tuple(mappings[pos][row[pos]] for pos in range(3))
                counts[key] = counts.get(key, 0) + 1
            if all(count >= k for count in counts.values()):
                loss = sum(hierarchies[['c1', 'c2', 'c3'][pos]]
                           ['levels'][combo[pos]]['loss']
                           for pos in range(3))
                feasible.append((combo, loss))
                if best_loss is None or loss < best_loss:
                    best_loss = loss
                    best_combo = combo
        return best_combo, best_loss, feasible

    for k in (1, 2, 3, 5, 9, 27):
        ref_combo, ref_loss, feasible = reference(k)
        if feasible:
            view = generalize(table, hierarchies, k=k)
            assert list(view.selected_levels.values()) == list(ref_combo)
            assert view.total_loss == ref_loss
            assert sum(row[-1] for row in list(view.groups)[1:]) == 27
        else:
            with pytest.raises(GeneralizationError):
                generalize(table, hierarchies, k=k).selected_levels


def test_generalize_tie_prefers_earlier_columns_stay_specific():
    table = [['c1', 'c2'],
             ['a', 'p'], ['a', 'q'], ['b', 'p'], ['b', 'q']]
    hierarchies = {
        'c1': {'levels': [
            {'loss': 0, 'mapping': {'a': 'a', 'b': 'b'}},
            {'loss': 1, 'mapping': {'a': 'g', 'b': 'g'}},
        ]},
        'c2': {'levels': [
            {'loss': 0, 'mapping': {'p': 'p', 'q': 'q'}},
            {'loss': 1, 'mapping': {'p': 'h', 'q': 'h'}},
        ]},
    }
    view = generalize(table, hierarchies, k=4)
    assert view.selected_levels == {'c1': 1, 'c2': 1}


def test_generalize_preserves_rows_order_and_other_fields():
    table = _table()
    view = generalize(table, _hierarchies(), k=2)
    out = list(view.table)
    assert len(out) == len(table)
    assert [row[0] for row in out[1:]] == [row[0] for row in table[1:]]
    assert [tuple(row[3:]) for row in out[1:]] == \
        [tuple(row[3:]) for row in table[1:]]
    assert out[0] == tuple(table[0])
    ieq(out, list(view.table))


def test_generalize_group_counts_include_duplicates():
    view = generalize(_table(), _hierarchies(), k=2)
    counts = {tuple(row[:-1]): row[-1] for row in list(view.groups)[1:]}
    assert sum(counts.values()) == 7
    assert all(count >= 2 for count in counts.values())


def test_generalize_tie_breaks_by_column_order():
    table = [
        ['a', 'b'],
        ['x', 'p'], ['x', 'q'],
        ['y', 'p'], ['y', 'q'],
    ]
    hierarchies = {
        'a': {'levels': [
            {'loss': 0, 'mapping': {'x': 'x', 'y': 'y'}},
            {'loss': 1, 'mapping': {'x': 'g', 'y': 'g'}},
        ]},
        'b': {'levels': [
            {'loss': 0, 'mapping': {'p': 'p', 'q': 'q'}},
            {'loss': 1, 'mapping': {'p': 'h', 'q': 'h'}},
        ]},
    }
    view = generalize(table, hierarchies, k=4)
    assert view.selected_levels == {'a': 1, 'b': 1}
    assert view.total_loss == 2


def test_generalize_infeasible_reports_reason_and_keeps_source():
    table = [
        ['a'], ['x'], ['y'], ['z'],
    ]
    hierarchies = {
        'a': {'levels': [
            {'loss': 0, 'mapping': {'x': 'x', 'y': 'y', 'z': 'z'}},
            {'loss': 1, 'mapping': {'x': 'g', 'y': 'g', 'z': 'h'}},
        ]},
    }
    with pytest.raises(GeneralizationError) as exc:
        generalize(table, hierarchies, k=5).selected_levels
    assert 'no feasible level combination' in str(exc.value)
    assert len(table) == 4


def test_generalize_uncovered_value_is_rejected():
    table = [['a'], ['x'], ['x'], ['w']]
    hierarchies = {
        'a': {'levels': [
            {'loss': 0, 'mapping': {'x': 'x'}},
        ]},
    }
    with pytest.raises(GeneralizationError) as exc:
        generalize(table, hierarchies, k=2).selected_levels
    assert 'not covered' in str(exc.value)


def test_generalize_none_keep_policy():
    table = [['a'], ['x'], ['x'], [None], [None]]
    hierarchies = {
        'a': {'levels': [
            {'loss': 0, 'mapping': {'x': 'x'}},
            {'loss': 1, 'mapping': {'x': 'z'}},
        ]},
    }
    view = generalize(table, hierarchies, k=2, none_policy='keep')
    out = [row[0] for row in list(view.table)[1:]]
    assert out == ['x', 'x', None, None]


def test_generalize_none_map_policy():
    table = [['a'], ['x'], ['x'], ['x'], [None], [None]]
    hierarchies = {
        'a': {'levels': [
            {'loss': 0, 'mapping': {'x': 'x', None: None}},
            {'loss': 1, 'mapping': {'x': 'g', None: 'n'}},
            {'loss': 2, 'mapping': {'g': 'any', 'n': 'any', None: 'any'}},
        ]},
    }
    view = generalize(table, hierarchies, k=2, none_policy='map')
    out = [row[0] for row in list(view.table)[1:]]
    assert out == ['x', 'x', 'x', None, None]
    mid_hier = {
        'a': {'levels': [
            {'loss': 0, 'mapping': {'x': 'x', None: None}},
            {'loss': 1, 'mapping': {'x': 'g', None: 'g'}},
        ]},
    }
    mid_table = [['a'], ['x'], ['x'], [None], [None]]
    mid = generalize(mid_table, mid_hier, k=3, none_policy='map')
    assert [row[0] for row in list(mid.table)[1:]] == \
        ['g', 'g', 'g', 'g']
    forced = generalize(table, hierarchies, k=4, none_policy='map')
    assert [row[0] for row in list(forced.table)[1:]] == \
        ['any', 'any', 'any', 'any', 'any']


def test_generalize_none_map_requires_explicit_none_key():
    table = [['a'], ['x'], ['x'], [None], [None]]
    hierarchies = {
        'a': {'levels': [
            {'loss': 0, 'mapping': {'x': 'x'}},
        ]},
    }
    with pytest.raises(GeneralizationError):
        generalize(table, hierarchies, k=2,
                   none_policy='map').selected_levels


def test_generalize_none_error_policy():
    table = [['a'], ['x'], ['x'], [None]]
    hierarchies = {
        'a': {'levels': [
            {'loss': 0, 'mapping': {'x': 'x'}},
        ]},
    }
    with pytest.raises(GeneralizationError) as exc:
        generalize(table, hierarchies, k=2,
                   none_policy='error').selected_levels
    assert 'None' in str(exc.value)


def test_generalize_rejects_splitting_level():
    hierarchies = {
        'a': {'levels': [
            {'loss': 0, 'mapping': {'x': 'x', 'y': 'y'}},
            {'loss': 1, 'mapping': {'x': 'g', 'y': 'g'}},
            {'loss': 2, 'mapping': {'g': 'u'}},
            {'loss': 3, 'mapping': {'u': 'u1', 'u1': 'u2'}},
        ]},
    }
    table = [['a'], ['x'], ['x'], ['y'], ['y']]
    with pytest.raises(GeneralizationError) as exc:
        generalize(table, hierarchies, k=2).selected_levels
    assert 'absent from the previous level' in str(exc.value)


def test_generalize_rejects_missing_upper_level_labels():
    hierarchies = {
        'a': {'levels': [
            {'loss': 0, 'mapping': {'x': 'x', 'y': 'y'}},
            {'loss': 1, 'mapping': {'x': 'g1', 'y': 'g2'}},
            {'loss': 2, 'mapping': {'g1': 'any'}},
        ]},
    }
    table = [['a'], ['x'], ['x'], ['y'], ['y']]
    with pytest.raises(GeneralizationError) as exc:
        generalize(table, hierarchies, k=2).selected_levels
    assert 'does not cover labels' in str(exc.value)


def test_generalize_rejects_mixed_type_boundary():
    hierarchies = {
        'a': {'levels': [
            {'loss': 0, 'mapping': {'x': 1, 'y': 's'}},
        ]},
    }
    table = [['a'], ['x'], ['x'], ['y'], ['y']]
    with pytest.raises(GeneralizationError) as exc:
        generalize(table, hierarchies, k=2).selected_levels
    assert 'mixes label types' in str(exc.value)


def test_generalize_allows_cross_level_type_change():
    table = [['a'], [1], [1], [2], [2]]
    hierarchies = {
        'a': {'levels': [
            {'loss': 0, 'mapping': {1: 1, 2: 2}},
            {'loss': 1, 'mapping': {1: 'low', 2: 'high'}},
            {'loss': 2, 'mapping': {'low': 'any', 'high': 'any'}},
        ]},
    }
    view = generalize(table, hierarchies, k=2)
    assert view.selected_levels == {'a': 0}
    assert [row[0] for row in list(view.table)[1:]] == [1, 1, 2, 2]
    forced = generalize(table, hierarchies, k=4)
    assert forced.selected_levels == {'a': 2}


def test_generalize_candidate_cap_is_enforced():
    table = [['a', 'b'], ['x', 'p'], ['x', 'p']]
    hierarchies = {
        'a': {'levels': [{'loss': 0, 'mapping': {'x': 'x'}},
                         {'loss': 1, 'mapping': {'x': 'z'}}]},
        'b': {'levels': [{'loss': 0, 'mapping': {'p': 'p'}},
                         {'loss': 1, 'mapping': {'p': 'r'}}]},
    }
    with pytest.raises(GeneralizationError) as exc:
        generalize(table, hierarchies, k=2,
                   max_candidates=3).selected_levels
    assert 'exceed max_candidates' in str(exc.value)


def test_generalize_spills_to_disk_and_reuses_file(tmp_path):
    table = [['id', 'a', 'payload']]
    for index in range(20):
        table.append([index, 'x' if index % 2 else 'y', index * 10])
    hierarchies = {
        'a': {'levels': [
            {'loss': 0, 'mapping': {'x': 'x', 'y': 'y'}},
            {'loss': 1, 'mapping': {'x': 'g', 'y': 'g'}},
        ]},
    }
    view = generalize(table, hierarchies, k=10, max_rows=5)
    assert view.spill_path is not None
    assert len(list(view.table)) == len(table)
    assert [row[2] for row in list(view.table)[1:]] == \
        [row[2] for row in table[1:]]
    assert sum(row[-1] for row in list(view.groups)[1:]) == 20
    view.close()
    import os
    assert not os.path.exists(view.spill_path or '')


def test_generalize_spilled_store_supports_concurrent_iteration():
    table = [['id', 'a']]
    for index in range(12):
        table.append([index, 'x' if index % 2 else 'y'])
    hierarchies = {
        'a': {'levels': [
            {'loss': 0, 'mapping': {'x': 'x', 'y': 'y'}},
            {'loss': 1, 'mapping': {'x': 'g', 'y': 'g'}},
        ]},
    }
    view = generalize(table, hierarchies, k=6, max_rows=4)
    assert view.spill_path is not None
    first = iter(view.table)
    second = iter(view.table)
    header_a = next(first)
    header_b = next(second)
    interleaved = []
    for row_a, row_b in zip(first, second):
        interleaved.append((row_a[0], row_b[0]))
    assert header_a == header_b == ('id', 'a')
    assert interleaved == [(index, index) for index in range(12)]
    view.close()


def test_generalize_reads_source_only_once():
    table = _table()
    consumed = {'count': 0}

    class OneShotTable(object):
        def __iter__(self):
            consumed['count'] += 1
            return iter(table)

    view = generalize(OneShotTable(), _hierarchies(), k=2)
    list(view.table)
    list(view.groups)
    list(view.table)
    assert consumed['count'] == 1


def test_generalize_fluent_and_header_validation():
    import petl as etl
    view = etl.wrap(_table()).generalize(_hierarchies(), k=2)
    assert etl.nrows(view.table) == 7
    with pytest.raises(GeneralizationError):
        etl.wrap(_table()).generalize(
            {'missing': _hierarchies()['age']}, k=2).selected_levels


def test_generalize_rejects_invalid_arguments():
    table = [['a'], ['x'], ['x']]
    hierarchy = {'a': {'levels': [{'loss': 0, 'mapping': {'x': 'x'}}]}}
    with pytest.raises(GeneralizationError):
        generalize(table, hierarchy, k=True).selected_levels
    with pytest.raises(GeneralizationError):
        generalize(table, hierarchy, k=0).selected_levels
    with pytest.raises(GeneralizationError):
        generalize(table, {}, k=2).selected_levels
    with pytest.raises(GeneralizationError):
        generalize([['a']], hierarchy, k=2).selected_levels
    with pytest.raises(GeneralizationError):
        generalize(iter([]), hierarchy, k=2).selected_levels
    with pytest.raises(GeneralizationError):
        bad_loss = {'a': {'levels': [
            {'loss': float('nan'), 'mapping': {'x': 'x'}}]}}
        generalize(table, bad_loss, k=2).selected_levels


def test_generalize_rejects_unhashable_labels():
    table = [['a'], ['x'], ['x']]
    hierarchy = {'a': {'levels': [{'loss': 0,
                                   'mapping': {'x': ['u']}}]}}
    with pytest.raises(GeneralizationError) as exc:
        generalize(table, hierarchy, k=2).selected_levels
    assert 'hashable' in str(exc.value)


def test_generalize_level_zero_must_be_identity():
    table = [['a'], ['x'], ['x'], ['y'], ['y']]
    hierarchy = {
        'a': {'levels': [
            {'loss': 0, 'mapping': {'x': 'g', 'y': 'g'}},
        ]},
    }
    with pytest.raises(GeneralizationError) as exc:
        generalize(table, hierarchy, k=2).selected_levels
    assert 'identity level' in str(exc.value)


def test_generalize_rejects_broken_type_boundary_between_levels():
    hierarchies = {
        'a': {'levels': [
            {'loss': 0, 'mapping': {1: 1, 2: 2}},
            {'loss': 1, 'mapping': {1: 'low', 2: 'high'}},
            {'loss': 2, 'mapping': {0: 'all', 1: 'all'}},
        ]},
    }
    table = [['a'], [1], [1], [2], [2]]
    with pytest.raises(GeneralizationError) as exc:
        generalize(table, hierarchies, k=4).selected_levels
    assert 'does not cover labels' in str(exc.value)


def test_generalize_disk_path_preserves_count_and_groups():
    table = [['id', 'a', 'payload']]
    for index in range(15):
        table.append([index, 'x' if index % 3 else 'y', 'p%02d' % index])
    hierarchies = {
        'a': {'levels': [
            {'loss': 0, 'mapping': {'x': 'x', 'y': 'y'}},
            {'loss': 1, 'mapping': {'x': 'g', 'y': 'g'}},
        ]},
    }
    view = generalize(table, hierarchies, k=5, max_rows=4)
    assert view.spill_path is not None
    assert len(list(view.table)) == 16
    payloads = [row[2] for row in list(view.table)[1:]]
    assert payloads == ['p%02d' % index for index in range(15)]
    assert sum(row[-1] for row in list(view.groups)[1:]) == 15
    view.close()
