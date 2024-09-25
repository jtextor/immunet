from enum import Enum, unique
import numpy as np
import json
from typing import List
from config import PANEL_FILE
from pathlib import Path


class EnumWithMetadata(Enum):
    @classmethod
    def list(cls):
        values = []
        for case in cls:
            values.append(case.value)
        return values


@unique
class MarkerExpression(EnumWithMetadata):
    positive = True
    negative = False
    wildcard = "*"


@unique
class SpecialPhenotypes(str, Enum):
    no = "No cell"
    other = "Other cell"
    invalid = "Invalid"

    def make_phenotype(self, markers_num):
        pheno = [MarkerExpression.negative] * markers_num
        return Phenotype(self.value, self.value, pheno, True)

    def make_anntype(self):
        return AnnotationType(self.value, self.value)


class AnnotationType:
    def __init__(self, main, detailed):
        self.main = main
        self.detailed = detailed

    @property
    def invalid(self):
        return self.main == SpecialPhenotypes.invalid.value


class Phenotype:
    def __init__(self,
                 main_type: str,
                 subtype: str,
                 phenotype: List[MarkerExpression] = None,
                 background: bool = False):
        self.main_type = main_type
        self.subtype = subtype
        self.phenotype = phenotype
        self.background = background

    @classmethod
    def from_dict(cls, dictionary, markers):
        if "type" not in dictionary:
            raise ValueError("Phenotype dictionary has to contain 'type' key")

        type = dictionary["type"]
        if not isinstance(type, str):
            raise ValueError("Type has to be a string")

        if "background" in dictionary:
            background = dictionary["background"]
            if not isinstance(background, bool):
                raise ValueError("Background property has to be boolean")
        else:
            background = False

        if background:
            bg_pheno = [MarkerExpression.negative] * len(markers)
            return Phenotype(type, type, bg_pheno, background)

        if "phenotype" not in dictionary:
            raise ValueError("Phenotype dictionary has to contain 'phenotype' key if a phenotype is foreground")

        phenotype_dict = dictionary["phenotype"]

        if not isinstance(phenotype_dict, dict):
            raise ValueError("Phenotype has to be defined as a dictionary of cellular marker expressions")

        phenotype = []
        for marker in markers:
            if marker not in phenotype_dict:
                raise ValueError(
                    f"Phenotype dictionary has to specify expression of every panel marker, the expression of {marker} is missing")

            expression_value = phenotype_dict[marker]
            if expression_value not in MarkerExpression.list():
                raise ValueError("Marker expression has to be false, true or '*'")

            expression = MarkerExpression(expression_value)
            phenotype.append(expression)

        if "subtype" in dictionary:
            subtype = dictionary["subtype"]
            if not isinstance(subtype, str):
                raise ValueError("Subtype has to be a string")

            return Phenotype(type, subtype, phenotype, background)
        else:
            return Phenotype(type, type, phenotype, background)

    def __eq__(self, other):
        if not isinstance(other, Phenotype):
            return False

        type_equal = self.main_type == other.main_type and self.subtype == other.subtype
        bg_equal = self.background == other.background
        pheno_equal = self.phenotype == other.phenotype
        return type_equal and bg_equal and pheno_equal


class Panel:
    def __init__(self,
                 name: str,
                 markers: List[str],
                 phenotypes: List[Phenotype]):
        self.name = name
        self.markers = markers
        self.phenotypes = phenotypes
        self.__annotation_types = None
        self.__decorable_types = None
        self.__main_types = None
        self.__foreground_types = None
        self.__background_types = None


    @classmethod
    def from_dict(cls, dictionary):
        if "panel" not in dictionary:
            raise ValueError("Panel dictionary has to contain 'panel' key")

        panel_name = dictionary["panel"]
        if not isinstance(panel_name, str):
            raise ValueError("Panel property has to be a string")

        if "markers" not in dictionary:
            raise ValueError("Panel dictionary has to contain 'markers' key")

        panel_markers = dictionary["markers"]
        if not isinstance(panel_markers, list) or len(panel_markers) == 0 or not all(isinstance(pm, str) for pm in panel_markers):
            raise ValueError("Markers property has to be a list of string values")

        if "phenotypes" not in dictionary:
            raise ValueError("Panel dictionary has to contain 'phenotypes' key")

        pheno_dicts = dictionary["phenotypes"]
        if not isinstance(pheno_dicts, list) or len(pheno_dicts) == 0 or not all(isinstance(pd, dict) for pd in pheno_dicts):
            raise ValueError("Phenotypes property has to be a list of dictionaries")

        phenotypes = [Phenotype.from_dict(pheno_dict, panel_markers) for pheno_dict in pheno_dicts]

        has_no_cell_type = any(pheno.main_type == SpecialPhenotypes.no for pheno in phenotypes)
        if not has_no_cell_type:
            phenotypes.append(SpecialPhenotypes.no.make_phenotype(len(panel_markers)))

        has_other_cell_type = any(pheno.main_type == SpecialPhenotypes.other for pheno in phenotypes)
        if not has_other_cell_type:
            phenotypes.append(SpecialPhenotypes.other.make_phenotype(len(panel_markers)))

        panel = cls(
            panel_name,
            panel_markers,
            phenotypes
        )
        return panel

    @property
    def annotation_types(self):
        if self.__annotation_types is None:
            self.__annotation_types = [pheno.subtype for pheno in self.phenotypes]

        return self.__annotation_types

    @property
    def main_types(self):
        if self.__main_types is None:
            self.__main_types = sorted(list(set([pheno.main_type for pheno in self.phenotypes])))

        return self.__main_types

    @property
    def decorable_types(self):
        if self.__decorable_types is None:
            decorable_types = np.unique([pheno.main_type for pheno in self.phenotypes if pheno.main_type != pheno.subtype])
            self.__decorable_types = decorable_types.tolist()

        return self.__decorable_types

    @property
    def foreground_types(self):
        if self.__foreground_types is None:
            self.__foreground_types = sorted(list(set([pheno.main_type for pheno in self.phenotypes if not pheno.background])))
        return self.__foreground_types

    @property
    def background_types(self):
        if self.__background_types is None:
            self.__background_types = sorted(list(set([pheno.main_type for pheno in self.phenotypes if pheno.background])))
        return self.__background_types

    def annotation_phenotype(self, type, positivity, likert_cutoff=3):
        if not isinstance(type, str):
            raise ValueError("type has to be a string")

        if not isinstance(likert_cutoff, int):
            raise ValueError("likert_cutoff has to be an integer")

        if not isinstance(positivity, list) or len(positivity) != len(self.markers) or not all(isinstance(p, int) for p in positivity):
            raise ValueError("positivity has to be a list of integers with length that matches a number of markers in a panel")

        if type not in self.main_types:
            return SpecialPhenotypes.invalid.make_anntype()
        elif type not in self.decorable_types:
            return AnnotationType(type, type)
        else:
            return self.decorable_annotation_phenotype(type, positivity, likert_cutoff)

    def __phenotype_for_binary_expression(self, binary_expression, candidate_phenotypes):
        matching_pheno = None
        for candidate_phenotype in candidate_phenotypes:
            pheno = candidate_phenotype.phenotype
            wildcard_inds = [i for i, e in enumerate(pheno) if e == MarkerExpression.wildcard]

            refined_pheno = np.delete(pheno, wildcard_inds)
            refined_pheno = [x.value for x in refined_pheno]
            refined_pos = np.delete(binary_expression, wildcard_inds).tolist()

            if refined_pheno == refined_pos:
                matching_pheno = candidate_phenotype
                break

        if matching_pheno != None:
            return AnnotationType(matching_pheno.main_type, matching_pheno.subtype)
        else:
            return SpecialPhenotypes.invalid.make_anntype()


    def decorable_annotation_phenotype(self, type, positivity, likert_cutoff):
        binary_positivity = [x > likert_cutoff for x in positivity]

        candidate_phenotypes = [pheno for pheno in self.phenotypes if pheno.main_type == type]
        return self.__phenotype_for_binary_expression(binary_positivity, candidate_phenotypes)


    def prediction_phenotype(self, predicted_pheno, activ_th=0.4):
        if not isinstance(activ_th, float) or activ_th < 0 or activ_th > 1:
            raise ValueError("activ_th has to be a float between 0 and 1")

        if not isinstance(predicted_pheno, list) or len(predicted_pheno) != len(self.markers) or not all(isinstance(p, float) for p in predicted_pheno):
            raise ValueError("predicted_pheno has to be a list of floats with length that matches a number of markers in a panel")

        binarised_prediction = [x > activ_th for x in predicted_pheno]

        # If no markers are predicted to be expressed but a cell was detected, it has 'Other cell' phenotypes
        if sum(binarised_prediction) == 0:
            return SpecialPhenotypes.other.make_anntype()

        return self.__phenotype_for_binary_expression(binarised_prediction, self.phenotypes)


def load_panels(relative_path=Path(PANEL_FILE)):
    panels = {}

    with open(relative_path) as f:
        panel_dicts = json.load(f)

    for panel_dict in panel_dicts:
        panel = Panel.from_dict(panel_dict)
        panels[panel.name] = panel

    return panels
