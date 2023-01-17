from abc import ABCMeta, abstractmethod
from enum import Enum, unique
import numpy as np


@unique
class Markers(str, Enum):
    cd3 = "cd3"
    foxp3 = "foxp3"
    cd20 = "cd20"
    cd45ro = "cd45ro"
    cd8 = "cd8"
    cd56 = "cd56"


@unique
class CellType(str, Enum):
    t = "T cell"
    b = "B cell"
    tumor = "Tumor cell"
    other = "Other cell"
    no = "No cell"
    invalid = "Invalid"  # cell type or annotated positivity or predicted phenotype do not make sense in the panel

    @property
    def main_type(self):
        return self.value

    @property
    def types(self):
        return [self]

    @property
    def desc(self):
        return self.value

    @property
    def is_invalid(self):
        return self == CellType.invalid


@unique
class TCellType(str, Enum):
    cytotoxic = "Cytotoxic"
    regulatory = "Regulatory"
    memory = "Memory"
    helper = "Helper"
    other = "Other"
    invalid = "Invalid"  # annotated positivity or predicted phenotype is impossible for T cell

    @property
    def parent_type(self):
        return CellType.t

    @property
    def main_type(self):
        return self.parent_type.value

    @property
    def types(self):
        return [self, self.parent_type]

    @property
    def desc(self):
        return " ".join((self.value, self.parent_type.desc))

    @property
    def is_invalid(self):
        return self == TCellType.invalid


class Cell:
    def __init__(self, type):
        self.type = type

    @property
    def name(self):
        return self.desc


class ForegroundCell(Cell):
    def __init__(self, type, expresses, panel_markers):
        super().__init__(type)
        self.expresses = expresses
        self.phenotype = [int(marker in expresses) for marker in panel_markers]


class DecorableCell(ForegroundCell):
    def __init__(self, type, expresses, panel_markers):
        super().__init__(type, expresses, panel_markers)
        self.parent_type = type.parent_type


@unique
class PanelType(str, Enum):
    lymphocyte = "lymphocyte"
    lymphocyte_nk_ro = "lymphocyte_nk_ro"


class Panel(metaclass=ABCMeta):
    def __init__(self, markers, fg_cells, bg_cells):
        self.markers = markers
        self.fg_cells = fg_cells
        self.bg_cells = bg_cells
        self.cells = fg_cells + bg_cells

    @property
    def cell_subtypes(self):
        return sorted([cell.name for cell in self.cells])

    @property
    def cell_types(self):
        return sorted(list(set([cell.type.main_type for cell in self.cells])))

    @property
    def fg_cell_types(self):
        return sorted(list(set([cell.type.main_type for cell in self.fg_cells])))

    @property
    def bg_cell_types(self):
        return sorted(list(set([cell.type.main_type for cell in self.bg_cells])))

    @property
    def decorable_cell_types(self):
        return sorted(
            list(
                set(
                    [
                        cell.type.main_type
                        for cell in self.cells
                        if isinstance(cell, DecorableCell)
                    ]
                )
            )
        )

    def positivity_for_cell_type(self, type):
        for cell in self.fg_cells:
            if cell.type == type:
                return cell.phenotype

        return None

    def cell_from_annotation(self, type, positivity):

        if type not in self.cell_types:
            return CellType.invalid
        elif type not in self.decorable_cell_types:
            return CellType(type)
        else:
            return self.decorable_cell_from_annotation(type, positivity)

    @abstractmethod
    def decorable_cell_from_annotation(self, type, positivity):
        raise NotImplementedError

    @abstractmethod
    def cell_from_prediction(
            self, phenotype, activ_th=0.4
    ):
        raise NotImplementedError


class LymPanel(Panel):
    def __init__(self):

        markers = [
            Markers.cd3,
            Markers.foxp3,
            Markers.cd20,
            Markers.cd45ro,
            Markers.cd8,
        ]

        cyt_T = DecorableCell(TCellType.cytotoxic, [Markers.cd3, Markers.cd8], markers)
        reg_T = DecorableCell(
            TCellType.regulatory, [Markers.cd3, Markers.foxp3], markers
        )
        help_T = DecorableCell(TCellType.helper, [Markers.cd3], markers)
        mem_T = DecorableCell(TCellType.memory, [Markers.cd3, Markers.cd45ro], markers)
        b_cell = ForegroundCell(CellType.b, [Markers.cd20], markers)
        tumour_cell = Cell(CellType.tumor)
        other_cell = Cell(CellType.other)
        no_cell = Cell(CellType.no)

        fg_cells = [cyt_T, reg_T, help_T, mem_T, b_cell]
        bg_cells = [tumour_cell, other_cell, no_cell]

        super().__init__(markers, fg_cells, bg_cells)

    def decorable_cell_from_annotation(self, type, positivity):

        if type != CellType.t:
            raise ValueError("Only T cell is decorable in this panel")

        # CD20 is annotated as expressed or neither of T cell markers are annotated as expressed
        if (positivity[2] > 3) or (positivity[0] < 4 and positivity[1] < 4 and positivity[3] < 4 and positivity[4] < 4):
            return TCellType.invalid
        # Only CD45RO marker is annotated as expressed
        elif positivity[3] > 3 and (positivity[0] < 4 and positivity[1] < 4 and positivity[4] < 4):
            return TCellType.memory
        # Helper T cell: CD3+ FOXP3- CD8-
        elif positivity[0] > 3 and (positivity[1] < 4 and positivity[4] < 4):
            return TCellType.helper
        elif positivity[1] > 3 or positivity[4] > 3:
            # FOXP3 and CD8 expression is the same, impossible to determine subtype
            if positivity[1] == positivity[4]:
                return TCellType.other
            else:
                return TCellType.regulatory if positivity[1] > positivity[4] else TCellType.cytotoxic
        else:
            return TCellType.invalid


    def cell_from_prediction(
        self, phenotype, activ_th=0.4
    ):

        # No expression is predicted
        if sum(np.array(phenotype) > activ_th) == 0:
            return CellType.other

        # CD20 and T
        if phenotype[2] > activ_th and (
            phenotype[0] > activ_th
            or phenotype[1] > activ_th
            or phenotype[3] > activ_th
            or phenotype[4] > activ_th
        ):
            return CellType.invalid

        if phenotype[2] > activ_th:
            return CellType.b
        elif (
                phenotype[3] > activ_th
                and phenotype[0] <= activ_th
                and phenotype[1] <= activ_th
                and phenotype[4] <= activ_th
            ):
            return TCellType.memory
        elif (
                phenotype[0] > activ_th
                and phenotype[1] <= activ_th
                and phenotype[4] <= activ_th
            ):
            return TCellType.helper
        elif phenotype[1] > activ_th and phenotype[4] > activ_th:
            return TCellType.invalid
        else:
            return TCellType.regulatory if phenotype[1] > phenotype[4] else TCellType.cytotoxic


class LymNkRoPanel(Panel):
    def __init__(self):

        markers = [
            Markers.cd3,
            Markers.foxp3,
            # This marker is not used in the paper
            Markers.cd56,
            Markers.cd45ro,
            Markers.cd8,
        ]

        cyt_T = DecorableCell(TCellType.cytotoxic, [Markers.cd3, Markers.cd8], markers)
        reg_T = DecorableCell(
            TCellType.regulatory, [Markers.cd3, Markers.foxp3], markers
        )
        help_T = DecorableCell(TCellType.helper, [Markers.cd3], markers)
        mem_T = DecorableCell(TCellType.memory, [Markers.cd3, Markers.cd45ro], markers)
        tumour_cell = Cell(CellType.tumor)
        other_cell = Cell(CellType.other)
        no_cell = Cell(CellType.no)

        fg_cells = [cyt_T, reg_T, help_T, mem_T]
        bg_cells = [tumour_cell, other_cell, no_cell]

        super().__init__(markers, fg_cells, bg_cells)

    def decorable_cell_from_annotation(self, type, positivity):

        if type != CellType.t:
            raise ValueError("Only T cell is decorable in this panel")

        # CD56 is annotated as expressed or neither of T cell markers are annotated as expressed
        if (positivity[2] > 3) or (
                positivity[0] < 4 and positivity[1] < 4 and positivity[3] < 4 and positivity[4] < 4):
            return TCellType.invalid
        # Only CD45RO marker is annotated as expressed
        elif positivity[3] > 3 and (positivity[0] < 4 and positivity[1] < 4 and positivity[4] < 4):
            return TCellType.memory
        # Helper T cell: CD3+ FOXP3- CD8-
        elif positivity[0] > 3 and (positivity[1] < 4 and positivity[4] < 4):
            return TCellType.helper
        elif positivity[1] > 3 or positivity[4] > 3:
            # FOXP3 and CD8 expression is the same, impossible to determine subtype
            if positivity[1] == positivity[4]:
                return TCellType.other
            else:
                return TCellType.regulatory if positivity[1] > positivity[4] else TCellType.cytotoxic
        else:
            return TCellType.invalid

    def cell_from_prediction(
        self, phenotype, activ_th=0.4
    ):

        # No expression is predicted
        if sum(np.array(phenotype) >= activ_th) == 0:
            return CellType.other

        # CD56 and T
        if phenotype[2] > activ_th and (
                phenotype[0] > activ_th
                or phenotype[1] > activ_th
                or phenotype[3] > activ_th
                or phenotype[4] > activ_th
        ):
            return CellType.invalid

        if phenotype[2] > activ_th:
            # CD56 marker is not used in the paper
            return CellType.invalid
        elif (
                phenotype[3] > activ_th
                and phenotype[0] <= activ_th
                and phenotype[1] <= activ_th
                and phenotype[4] <= activ_th
            ):
            return TCellType.memory
        elif (
                phenotype[0] > activ_th
                and phenotype[1] <= activ_th
                and phenotype[4] <= activ_th
            ):
            return TCellType.helper
        elif phenotype[1] > activ_th and phenotype[4] > activ_th:
            return TCellType.invalid
        else:
            return TCellType.regulatory if phenotype[1] > phenotype[4] else TCellType.cytotoxic


panels = {
    PanelType.lymphocyte: LymPanel(),
    PanelType.lymphocyte_nk_ro: LymNkRoPanel()
}
