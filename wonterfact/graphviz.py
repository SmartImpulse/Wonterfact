# ----------------------------------------------------------------------------
# Copyright 2020 Benoit Fuentes <bf@benoit-fuentes.fr>
#
# This file is part of Wonterfact.
#
# Wonterfact is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# Wonterfact is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Wonterfact. If not, see <https://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------

"""Module to draw operation tree of wonterfact models"""

# Python System imports
import re
from pathlib import Path

# Relative imports
from . import core_nodes, root, observers, operators, utils, buds

# Third-party imports
import numpy as np
import graphviz
import string


def _get_node_shape(node):
    if isinstance(node, observers._Observer):
        return "doublecircle"
    if isinstance(node, root.Root):
        return "none"
    return "ellipse"


def _get_node_prefix(node, legend_dict, **extra_param):
    if isinstance(node, operators.Multiplier):
        if node.conv_idx_ids:
            conv_idx = "".join(
                [
                    legend_dict[idx]["letter"]
                    for idx in node.index_id[::-1]
                    if idx in node.conv_idx_ids
                ]
            )
            # return "&#8859;<sub><i>{}</i></sub>".format(conv_idx)
            return "(&lowast;)<sub><i>{}</i></sub>".format(conv_idx)
        # return "&#215;"
        # return "&Pi;"
        return "(&times;)"

    if isinstance(node, operators.Multiplexer):
        if node.multiplexer_idx is not None:
            return "(&#8741;<sub><i>{}</i></sub>)".format(
                legend_dict[node.multiplexer_idx]["letter"]
            )
        return "(&#8741;)"

    if isinstance(node, operators.Adder):
        # return "&#43;"
        # return "&Sigma;"
        return "(+)"

    if isinstance(node, observers.PosObserver):
        integer_observations = extra_param.get("integer_observations", False)
        if integer_observations:
            return "&#8469;"
        return "&#8477;<sup>+</sup>"

    if isinstance(node, observers.RealObserver):
        integer_observations = extra_param.get("integer_observations", False)
        if integer_observations:
            return "&#8484;"
        return "&#8477;"

    if isinstance(node, operators.Integrator):
        return "&#8747;<sub><i>{}</i></sub>".format(
            legend_dict[node.index_id[-1]]["letter"]
        )

    if isinstance(node, operators.Proxy):
        return "(=)"

    return ""


def _get_edge_label(node, child, legend_dict):
    if not isinstance(node, core_nodes._DynNodeData):
        return None
    label = ""
    slice_for_child = node.slicing_for_children_dict[child]
    if slice_for_child != Ellipsis:
        explicit_slice_raw = utils.explicit_slice(slice_for_child, node.tensor.ndim)
        explicit_slice = []
        masked_idx = []
        num_idx = 0
        for elem in explicit_slice_raw:
            elem_as_np = utils._is_bool_masking(elem)
            if elem_as_np is not None:
                masked_idx += node.index_id[num_idx : num_idx + elem_as_np.ndim]
                num_idx += elem_as_np.ndim
                explicit_slice += [slice(None),] * elem_as_np.ndim
            else:
                explicit_slice.append(elem)
                num_idx += 1

        for idx, sl, dim_axis in zip(
            node.index_id[::-1], explicit_slice[::-1], node.tensor.shape[::-1]
        ):
            if sl != slice(None):
                letter = legend_dict[idx]["letter"]
                if isinstance(sl, int):
                    label += "<i>{}</i>={};".format(letter, sl)
                elif isinstance(sl, slice):
                    start = "" if sl.start in [0, None] else sl.start
                    step = "" if sl.step in [1, None] else sl.step
                    stop = "" if sl.stop in [dim_axis, None] else sl.stop
                    label += "<i>{}</i>={}:{}:{};".format(letter, start, stop, step)
                else:
                    label += "<i>{}</i>=["
                    label += ",".join(str(ax) for ax in sl)
                    label += "]"
        if masked_idx:
            masked_letters = "".join(
                [legend_dict[idx]["letter"] for idx in masked_idx[::-1]]
            )
            label += "mask: " + masked_letters
    if node.strides_for_children_dict[child]:
        label += "strides: {}".format(node.strides_for_children_dict[child])
    index_id_for_child = node.get_index_id_for_children(child)
    if index_id_for_child != node.index_id:
        label_new_idx = _insert_given_symbol(
            index_id_for_child,
            node.get_norm_axis_for_children(child),
            node.get_tensor_for_children(child).ndim,
            legend_dict,
        )
        # label_new_idx = '&rarr;' + label_new_idx
        if label:
            label += "<br/>" + label_new_idx
        else:
            label = label_new_idx
    return label or None


def _draw_tree(
    tree,
    fileformat=None,
    filename=None,
    legend_dict=None,
    prior_nodes=False,
    view=True,
    show_node_names=False,
    integer_observations=False,
):
    if tree.current_iter == 0:
        tree._first_iteration(check_model_validity=False)
    fileformat = fileformat or "pdf"
    filename = filename or tree.name
    filename = Path(filename)
    suffix = filename.suffix
    if suffix:
        fileformat = suffix[1:]
    filename = filename.with_suffix("")
    legend_dict = legend_dict or {}
    all_index_id = set.union(
        *(set(node.index_id) for node in tree.census() if hasattr(node, "index_id"))
    )
    used_letters = set()
    for elem in legend_dict.values():
        letter = elem.get("letter", None)
        if letter is not None:
            used_letters.add(letter[0])
    for index_id in all_index_id:
        if not index_id in legend_dict or "letter" not in legend_dict[index_id]:
            idx_id = index_id if isinstance(index_id, str) else ""
            idx_id2 = re.sub("[^a-z]+", "", idx_id)
            letter = next(
                (
                    let
                    for let in idx_id2 + string.ascii_lowercase
                    if let not in used_letters
                ),
                None,
            )
            if letter is None:
                raise ValueError(
                    "Not enough letters in the alphabet. Please provide"
                    "`legend_dict` with your own letters with subscripts"
                    "to represent each index_id"
                )
            if not index_id in legend_dict:
                legend_dict[index_id] = {}
            legend_dict[index_id]["letter"] = letter
            used_letters.add(letter[0])
            if "description" not in legend_dict[index_id] and letter != index_id:
                legend_dict[index_id]["description"] = index_id

    graph = graphviz.Digraph(
        name=tree.name, format=fileformat, filename=filename, engine="dot"
    )
    graph.attr("node", color="#0b51c3f2", fontname="Times-Roman", height="0")
    graph.attr("edge", color="#0b51c3f2", arrowhead="none", fontname="Times-Roman")
    for node in tree.census():
        xlabel = _html(_small_font(node.name)) if show_node_names else None

        # special form for root
        if node is tree:
            # node_label = """<<font  COLOR="#0b51c3f2"><o>/ / / / / /</o></font>>"""
            # node_label = '<<font  COLOR="#0b51c3f2">___<br/>__<br/>_</font>>'
            node_label = (
                """<<table border="0" cellspacing="0" cellborder="1">"""
                """<tr>"""
                """<td width="9" height="9" fixedsize="true" sides="t"></td>"""
                """<td width="9" height="9" fixedsize="true" sides="tb"></td>"""
                """<td width="9" height="9" fixedsize="true" sides="tb"></td>"""
                """<td width="9" height="9" fixedsize="true" sides="tb"></td>"""
                """<td width="9" height="9" fixedsize="true" sides="t"></td>"""
                """</tr>"""
                """<tr>"""
                """<td width="9" height="9" fixedsize="true" style="invis"></td>"""
                """<td width="9" height="9" fixedsize="true" style="invis"></td>"""
                """<td width="9" height="9" fixedsize="true" sides="b"></td>"""
                """<td width="9" height="9" fixedsize="true" style="invis"></td>"""
                """<td width="9" height="9" fixedsize="true" style="invis"></td>"""
                """</tr>"""
                """</table>>"""
            )
            # node_label = ""
            graph.node(
                str(id(node)),
                label=node_label,
                shape="plain",
                peripheries="0",
                xlabel=xlabel,
                forcelabels="true",
                # image="/home/fuentes/Projects/wonterfact/wonterfact/images/ground.svg",
                # shape="epsf",
                # shapefile="/home/fuentes/Projects/wonterfact/wonterfact/images/ground.ps",
            )
        elif isinstance(node, observers.BlindObs):
            node_label = _make_node_label("", "?", True)
            with graph.subgraph(name="observers") as subg:
                subg.attr(rank="same")
                subg.node(
                    str(id(node)),
                    label=node_label,
                    shape="ellipse",
                    # shape="doublecircle",
                    peripheries="2",
                    style="diagonals",
                    xlabel=xlabel,
                    forcelabels="true",
                )
            # graph.node(
            #     str(id(node)),
            #     label=node_label,
            #     shape="ellipse",
            #     style="diagonals",
            #     peripheries="2",
            #     xlabel=xlabel,
            # )
        # all nodes except root
        else:
            # let us compute node_label
            if not node.index_id:
                node_label = _make_node_label(
                    "", "&middot;", underline=node.tensor_has_energy
                )
            else:
                if node.tensor_has_energy or node.level == 0:
                    index_label = "".join(
                        [legend_dict[idx]["letter"] for idx in node.index_id[::-1]]
                    )
                    index_label = _italic(index_label)
                    underline = node.tensor_has_energy
                else:
                    index_label = _insert_given_symbol(
                        node.index_id, node.norm_axis, node.tensor.ndim, legend_dict
                    )
                    underline = False
                node_prefix = _get_node_prefix(
                    node, legend_dict, integer_observations=integer_observations
                )
                node_prefix = _small_font(node_prefix)
                node_label = _make_node_label(node_prefix, index_label, underline)

            # special shape for observers
            if isinstance(node, observers._Observer):
                with graph.subgraph(name="observers") as subg:
                    subg.attr(rank="same")
                    subg.node(
                        str(id(node)),
                        label=node_label,
                        shape="ellipse",
                        # shape="doublecircle",
                        peripheries="2",
                        style="diagonals",
                        xlabel=xlabel,
                        forcelabels="true",
                    )
            # special shape for hyperparameter buds
            elif isinstance(node, buds._Bud):
                if hasattr(node, "update_period") and node.update_period == 0:
                    peripheries = "2"
                else:
                    peripheries = "1"
                if prior_nodes:
                    graph.node(
                        str(id(node)),
                        label=node_label,
                        shape="box",
                        peripheries=peripheries,
                        # style="diagonals",
                        xlabel=xlabel,
                        forcelabels="true",
                    )
            # leaves and operators
            else:
                if hasattr(node, "update_period") and node.update_period == 0:
                    peripheries = "2"
                else:
                    peripheries = "1"
                graph.node(
                    str(id(node)),
                    label=node_label,
                    shape="ellipse",
                    peripheries=peripheries,
                    xlabel=xlabel,
                    forcelabels="true",
                )
    for node in tree.census():
        if node != tree:
            if node.level != 0 or prior_nodes:
                for child in node.list_of_children:
                    edge_label = _get_edge_label(node, child, legend_dict)
                    if edge_label:
                        edge_label = _html(_small_font(edge_label))
                    graph.edge(str(id(node)), str(id(child)), taillabel=edge_label)

    if any("description" in idx_dict for idx_dict in legend_dict.values()):
        label_legend = (
            '<TABLE border="0">'
            '<TR><TD colspan="2" align="left" cellpadding="0">'
            "<b>Indexes</b> </TD></TR>"
        )
        for idx_dict in legend_dict.values():
            if "description" in idx_dict:
                label_legend += (
                    '<TR><TD align="left" cellpadding="0"><i>{}</i>:</TD>'
                    '<TD align="left"  cellpadding="0">{}<i> </i></TD>'
                    "</TR>".format(idx_dict["letter"], idx_dict["description"])
                )
        label_legend = label_legend + "</TABLE>"
        with graph.subgraph(name="observers") as subgraph:
            subgraph.attr(rank="same")
            subgraph.node(
                "legend",
                label=_html(label_legend),
                shape="box",
                # fontsize='8',
                style="dotted",
            )

    if view:
        graph.view(cleanup=True)
    else:
        graph.render(cleanup=True)
    return graph


def _html(input_str):
    return "<" + _clean_html(_detect_sub(input_str)) + ">"


def _clean_html(input_str):
    input_str = "".join(input_str.split("<i></i>"))
    input_str = "".join(input_str.split("<u></u>"))
    return input_str


def _italic(input_str):
    output_str = "<i>" + input_str + "</i>"
    return output_str


def _underline(input_str):
    return "<u>" + _italic(input_str) + "</u>"


def _small_font(input_str, fontsize=8):
    if not input_str:
        return input_str
    return '<font point-size="{}">'.format(fontsize) + input_str + "</font>"


def _detect_sub(input_str):
    def change_sub(input_str):
        return "</i><sub>" + input_str.group()[2:-1] + "</sub><i>"

    return re.sub("_{[^}]+}", change_sub, input_str)


def _insert_given_symbol(index_id, norm_axis, ndim, legend_dict):
    norm_axis = range(ndim) if norm_axis is None else norm_axis
    len_norm = len(norm_axis)
    if set(norm_axis) == set(range(ndim - len_norm, ndim)):
        index_id_list = index_id
    else:
        index_id_list = [
            idx for num_idx, idx in enumerate(index_id) if num_idx not in norm_axis
        ] + [idx for num_idx, idx in enumerate(index_id) if num_idx in norm_axis]
    index_label = "".join([legend_dict[idx]["letter"] for idx in index_id_list[::-1]])
    if len_norm < len(index_id_list):
        index_label = (
            _italic(index_label[:len_norm]) + " |" + _italic(index_label[len_norm:])
        )
    else:
        index_label = _italic(index_label)
    if len_norm == 0:
        index_label = "&middot;" + index_label
    return index_label


def _make_node_label(prefix, suffix, underline=False):
    if underline:
        underline_str = ' sides="b"'
    else:
        underline_str = ' border="0"'
    if prefix:
        first_col = '<td border="0">{}  </td>'.format(prefix)
    else:
        first_col = ""
    node_label = (
        '<table border="0" cellspacing="0" cellborder="1" CELLPADDING="0" COLOR="black">'
        '<tr border="0">'
        "{}<td{}>{}</td>"
        "</tr>"
        "</table>".format(first_col, underline_str, suffix)
    )
    return _html(node_label)
