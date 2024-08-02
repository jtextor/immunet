import json
import unittest
from unittest import TestCase
from pathlib import Path
from immunet.panels import SpecialPhenotypes, MarkerExpression, AnnotationType, Phenotype, Panel


def get_test_panel_dics():
    test_panels_fp = Path("data/panels.json")

    with open(test_panels_fp) as f:
        panel_dicts = json.load(f)

    return panel_dicts

class TestSpecialPhenotypes(TestCase):
    def test_make_phenotype(self):
        markers_num = 4
        expected_phenotype = [MarkerExpression.negative] * markers_num

        # Check the expected values for each special phenotype
        no_phenotype = SpecialPhenotypes.no.make_phenotype(markers_num)
        self.assertEqual(no_phenotype.main_type, "No cell")
        self.assertEqual(no_phenotype.subtype, "No cell")
        self.assertTrue(no_phenotype.background)
        self.assertEqual(no_phenotype.phenotype, expected_phenotype)

        other_phenotype = SpecialPhenotypes.other.make_phenotype(markers_num)
        self.assertEqual(other_phenotype.main_type, "Other cell")
        self.assertEqual(other_phenotype.subtype, "Other cell")
        self.assertTrue(other_phenotype.background)
        self.assertEqual(other_phenotype.phenotype, expected_phenotype)

        invalid_phenotype = SpecialPhenotypes.invalid.make_phenotype(markers_num)
        self.assertEqual(invalid_phenotype.main_type, "Invalid")
        self.assertEqual(invalid_phenotype.subtype, "Invalid")
        self.assertTrue(invalid_phenotype.background)
        self.assertEqual(invalid_phenotype.phenotype, expected_phenotype)

        # Check weird number of markers
        markers_num = 0
        expected_phenotype = [MarkerExpression.negative] * markers_num

        no_phenotype = SpecialPhenotypes.no.make_phenotype(markers_num)
        self.assertEqual(no_phenotype.phenotype, expected_phenotype)

        markers_num = 1
        expected_phenotype = [MarkerExpression.negative] * markers_num

        no_phenotype = SpecialPhenotypes.no.make_phenotype(markers_num)
        self.assertEqual(no_phenotype.phenotype, expected_phenotype)

        markers_num = 100
        expected_phenotype = [MarkerExpression.negative] * markers_num

        no_phenotype = SpecialPhenotypes.no.make_phenotype(markers_num)
        self.assertEqual(no_phenotype.phenotype, expected_phenotype)

    def test_make_anntype(self):
        # Check the expected values for each special phenotype
        no_anntype = SpecialPhenotypes.no.make_anntype()
        self.assertEqual(no_anntype.main, "No cell")
        self.assertEqual(no_anntype.detailed, "No cell")

        other_anntype = SpecialPhenotypes.other.make_anntype()
        self.assertEqual(other_anntype.main, "Other cell")
        self.assertEqual(other_anntype.detailed, "Other cell")

        invalid_anntype = SpecialPhenotypes.invalid.make_anntype()
        self.assertEqual(invalid_anntype.main, "Invalid")
        self.assertEqual(invalid_anntype.detailed, "Invalid")


class TestAnnotationType(TestCase):
    def test_invalid(self):
        annotation_type = AnnotationType("T cell", "Tcyt")
        self.assertFalse(annotation_type.invalid)

        annotation_type = AnnotationType("Invalid", "Invalid")
        self.assertTrue(annotation_type.invalid)


class TestPhenotype(TestCase):
    def test_from_dict_exceptions(self):
        markers = ["M1", "M2", "M3", "M4", "M5"]
        # Test exceptions
        dictionary = {}
        with self.assertRaisesRegex(ValueError, "Phenotype dictionary has to contain 'type' key"):
            Phenotype.from_dict(dictionary, markers)

        dictionary = {"cell": "T cell"}
        with self.assertRaisesRegex(ValueError, "Phenotype dictionary has to contain 'type' key"):
            Phenotype.from_dict(dictionary, markers)

        dictionary = {"type": 1}
        with self.assertRaisesRegex(ValueError, "Type has to be a string"):
            Phenotype.from_dict(dictionary, markers)

        dictionary = {"type": "T cell", "background": 2}
        with self.assertRaisesRegex(ValueError, "Background property has to be boolean"):
            Phenotype.from_dict(dictionary, markers)

        dictionary = {"type": "T cell"}
        with self.assertRaisesRegex(ValueError,
                                    "Phenotype dictionary has to contain 'phenotype' key if a phenotype is foreground"):
            Phenotype.from_dict(dictionary, markers)

        dictionary = {"type": "T cell", "phenotype": "Thelp"}
        with self.assertRaisesRegex(ValueError,
                                    "Phenotype has to be defined as a dictionary of cellular marker expressions"):
            Phenotype.from_dict(dictionary, markers)

        dictionary = {"type": "T cell", "phenotype": {"M1": False}}
        with self.assertRaisesRegex(ValueError,
                                    "Phenotype dictionary has to specify expression of every panel marker, the expression of [\w\d]+ is missing"):
            Phenotype.from_dict(dictionary, markers)

        dictionary = {"type": "T cell", "phenotype": {"M1": "&", "M2": False, "M3": True, "M4": False, "M5": False},
                      "subtype": False}
        with self.assertRaisesRegex(ValueError, "Marker expression has to be false, true or '*'"):
            Phenotype.from_dict(dictionary, markers)

        dictionary = {"type": "T cell", "phenotype": {"M1": False, "M2": False, "M3": True, "M4": False, "M5": False},
                      "subtype": False}
        with self.assertRaisesRegex(ValueError,
                                    "Subtype has to be a string"):
            Phenotype.from_dict(dictionary, markers)

    def test_from_dict_success(self):
        markers = ["M1", "M2", "M3", "M4", "M5"]

        # Test that when a valid dictionary describing phenotype is provided, the object is initialized correctly
        # Background
        dictionary = {"type": "Tumor cell", "background": True}
        expected_expresison = [MarkerExpression.negative] * len(markers)
        phenotype = Phenotype.from_dict(dictionary, markers)
        self.assertEqual(phenotype.main_type, dictionary["type"])
        self.assertEqual(phenotype.subtype, dictionary["type"])
        self.assertTrue(phenotype.background)
        self.assertEqual(phenotype.phenotype, expected_expresison)

        # Marker expression specified for a background annotation is ignored
        dictionary = {"type": "Tumor cell", "background": True,
                      "phenotype": {"M1": "*", "M2": False, "M3": True, "M4": False, "M5": False}}
        expected_expresison = [MarkerExpression.negative] * len(markers)
        phenotype = Phenotype.from_dict(dictionary, markers)
        self.assertEqual(phenotype.main_type, dictionary["type"])
        self.assertEqual(phenotype.subtype, dictionary["type"])
        self.assertTrue(phenotype.background)
        self.assertEqual(phenotype.phenotype, expected_expresison)

        # Foreground without subtype
        dictionary = {"type": "B cell", "phenotype": {"M1": "*", "M2": False, "M3": True, "M4": False, "M5": False}}
        expected_expresison = [MarkerExpression.wildcard, MarkerExpression.negative, MarkerExpression.positive,
                               MarkerExpression.negative, MarkerExpression.negative]
        phenotype = Phenotype.from_dict(dictionary, markers)
        self.assertEqual(phenotype.main_type, dictionary["type"])
        self.assertEqual(phenotype.subtype, dictionary["type"])
        self.assertFalse(phenotype.background)
        self.assertEqual(phenotype.phenotype, expected_expresison)

        # Foreground with subtype
        dictionary = {"type": "T cell", "subtype": "Tcyt",
                      "phenotype": {"M1": "*", "M2": False, "M3": True, "M4": False, "M5": False}}
        expected_expresison = [MarkerExpression.wildcard, MarkerExpression.negative, MarkerExpression.positive,
                               MarkerExpression.negative, MarkerExpression.negative]
        phenotype = Phenotype.from_dict(dictionary, markers)
        self.assertEqual(phenotype.main_type, dictionary["type"])
        self.assertEqual(phenotype.subtype, dictionary["subtype"])
        self.assertFalse(phenotype.background)
        self.assertEqual(phenotype.phenotype, expected_expresison)


class TestPanel(TestCase):
    def test_from_dict_exceptions(self):
        # Test exceptions
        panel_dict = {}
        with self.assertRaisesRegex(ValueError, "Panel dictionary has to contain 'panel' key"):
            Panel.from_dict(panel_dict)

        panel_dict = {"foo": "bar", "alice": 1}
        with self.assertRaisesRegex(ValueError, "Panel dictionary has to contain 'panel' key"):
            Panel.from_dict(panel_dict)

        panel_dict = {"panel": 1}
        with self.assertRaisesRegex(ValueError, "Panel property has to be a string"):
            Panel.from_dict(panel_dict)

        panel_dict = {"panel": "my_panel"}
        with self.assertRaisesRegex(ValueError, "Panel dictionary has to contain 'markers' key"):
            Panel.from_dict(panel_dict)

        panel_dict = {"panel": "my_panel", "markers": []}
        with self.assertRaisesRegex(ValueError, "Markers property has to be a list of string values"):
            Panel.from_dict(panel_dict)

        panel_dict = {"panel": "my_panel", "markers": [1, "M2", 3]}
        with self.assertRaisesRegex(ValueError, "Markers property has to be a list of string values"):
            Panel.from_dict(panel_dict)

        panel_dict = {"panel": "my_panel", "markers": ["M1", True]}
        with self.assertRaisesRegex(ValueError, "Markers property has to be a list of string values"):
            Panel.from_dict(panel_dict)

        panel_dict = {"panel": "my_panel", "markers": ["M1", "M2"]}
        with self.assertRaisesRegex(ValueError, "Panel dictionary has to contain 'phenotypes' key"):
            Panel.from_dict(panel_dict)

        panel_dict = {"panel": "my_panel", "markers": ["M1", "M2"], "phenotypes": "T cell, B cell"}
        with self.assertRaisesRegex(ValueError, "Phenotypes property has to be a list of dictionaries"):
            Panel.from_dict(panel_dict)

        panel_dict = {"panel": "my_panel", "markers": ["M1", "M2"], "phenotypes": []}
        with self.assertRaisesRegex(ValueError, "Phenotypes property has to be a list of dictionaries"):
            Panel.from_dict(panel_dict)

        panel_dict = {"panel": "my_panel", "markers": ["M1", "M2"], "phenotypes": ["M1", "M2"]}
        with self.assertRaisesRegex(ValueError, "Phenotypes property has to be a list of dictionaries"):
            Panel.from_dict(panel_dict)

        panel_dict = {"panel": "my_panel", "markers": ["M1", "M2"], "phenotypes": [{"type": "Tumor cell", "background": True}, "B cell"]}
        with self.assertRaisesRegex(ValueError, "Phenotypes property has to be a list of dictionaries"):
            Panel.from_dict(panel_dict)

    def test_from_dict_success(self):
        # Test that when initialised from dictionary, correct values are assigned
        panel_dicts = get_test_panel_dics()

        # Panel dictionary with no background phenotypes specified
        panel_dict_no_bg = panel_dicts[0]
        panel_no_bg = Panel.from_dict(panel_dict_no_bg)
        self.assertEqual(panel_no_bg.name, panel_dict_no_bg["panel"])
        expected_markers = panel_dict_no_bg["markers"]
        self.assertEqual(panel_no_bg.markers, expected_markers)
        specified_phenotypes = [Phenotype.from_dict(d, expected_markers) for d in panel_dict_no_bg["phenotypes"]]
        # If No / Other cell phenotypes are not specified they are added during the initialisation
        marker_num = len(expected_markers)
        bg_phenotypes = [SpecialPhenotypes.no.make_phenotype(marker_num), SpecialPhenotypes.other.make_phenotype(marker_num)]
        expected_phenotypes = specified_phenotypes + bg_phenotypes
        self.assertEqual(panel_no_bg.phenotypes, expected_phenotypes)

        # Panel dictionary with no background phenotypes specified
        panel_dict_bg = panel_dicts[1]
        panel_bg = Panel.from_dict(panel_dict_bg)
        self.assertEqual(panel_bg.name, panel_dict_bg["panel"])
        expected_markers = panel_dict_bg["markers"]
        self.assertEqual(panel_bg.markers, expected_markers)
        expected_phenotypes = [Phenotype.from_dict(d, expected_markers) for d in panel_dict_bg["phenotypes"]]
        self.assertEqual(panel_bg.phenotypes, expected_phenotypes)

    def test_annotation_types(self):
        # Test that annotation types list contains main types for phenotypes without a subtype and subtypes otherwise
        panel_dicts = get_test_panel_dics()

        panel_dict = panel_dicts[1]
        panel = Panel.from_dict(panel_dict)
        self.assertEqual(panel.annotation_types, ["Thelp", "Treg", "Tcyt", "Tmemory", "B cell", "Tumor cell", "No cell", "Other cell"])

    def test_main_types(self):
        # Test that main types list returns main types for all phenotypes with and without a subtype
        panel_dicts = get_test_panel_dics()

        panel_dict = panel_dicts[1]
        panel = Panel.from_dict(panel_dict)
        self.assertEqual(panel.main_types, ["B cell", "No cell", "Other cell", "T cell", "Tumor cell"])

    def test_decorable_types(self):
        # Test that decorable types list returns main types for phenotypes that have a subtype
        panel_dicts = get_test_panel_dics()

        panel_dict = panel_dicts[1]
        panel = Panel.from_dict(panel_dict)
        self.assertEqual(panel.decorable_types, ["T cell"])

        panel_dict = panel_dicts[2]
        panel = Panel.from_dict(panel_dict)
        self.assertEqual(sorted(panel.decorable_types), ["Dendritic cell", "T cell"])

    def test_foreground_types(self):
        # Test that foreground types list returns main types for foreground phenotypes
        panel_dicts = get_test_panel_dics()

        panel_dict = panel_dicts[1]
        panel = Panel.from_dict(panel_dict)
        self.assertEqual(panel.foreground_types, ["B cell", "T cell"])

        panel_dict = panel_dicts[2]
        panel = Panel.from_dict(panel_dict)
        self.assertEqual(panel.foreground_types, ["B cell", "Dendritic cell", "T cell"])

    def test_background_types(self):
        # Test that background types list returns main types for foreground phenotypes
        panel_dicts = get_test_panel_dics()

        panel_dict = panel_dicts[1]
        panel = Panel.from_dict(panel_dict)
        self.assertEqual(panel.background_types, ["No cell", "Other cell", "Tumor cell"])

        panel_dict = panel_dicts[2]
        panel = Panel.from_dict(panel_dict)
        self.assertEqual(panel.background_types, ["No cell", "Other cell"])

    def test_annotation_phenotype(self):
        panel_dicts = get_test_panel_dics()

        panel_dict = panel_dicts[0]
        panel = Panel.from_dict(panel_dict)

        # Check that expected exceptions are risen
        with self.assertRaisesRegex(ValueError, "type has to be a string"):
            panel.annotation_phenotype(0, [1, 1, 1, 1, 1])

        with self.assertRaisesRegex(ValueError, "likert_cutoff has to be an integer"):
            panel.annotation_phenotype("B cell", [1, 1, 1, 1, 1], 0.11)

        with self.assertRaisesRegex(ValueError, "positivity has to be a list of integers with length that matches a number of markers in a panel"):
            panel.annotation_phenotype("B cell", "CD20")

        with self.assertRaisesRegex(ValueError, "positivity has to be a list of integers with length that matches a number of markers in a panel"):
            panel.annotation_phenotype("B cell", [])

        with self.assertRaisesRegex(ValueError, "positivity has to be a list of integers with length that matches a number of markers in a panel"):
            panel.annotation_phenotype("B cell", [1, 1, 5])

        # Annotation without subtype
        type = "B cell"
        positivity = [1, 1, 5, 1, 1]
        ann_type = panel.annotation_phenotype(type, positivity)
        self.assertEqual(ann_type.main, type)
        self.assertEqual(ann_type.detailed, type)

        # Annotation without subtype, positivity is ignored
        type = "B cell"
        positivity = [5, 5, 5, 1, 1]
        ann_type = panel.annotation_phenotype(type, positivity)
        self.assertEqual(ann_type.main, type)
        self.assertEqual(ann_type.detailed, type)

        # Background annotation
        type = "No cell"
        positivity = [1, 1, 1, 1, 1]
        ann_type = panel.annotation_phenotype(type, positivity)
        self.assertEqual(ann_type.main, type)
        self.assertEqual(ann_type.detailed, type)

        # Annotation with subtype
        type = "T cell"
        positivity = [5, 1, 1, 1, 1]
        ann_type = panel.annotation_phenotype(type, positivity)
        self.assertEqual(ann_type.main, type)
        self.assertEqual(ann_type.detailed, "Thelp")

        # Annotation with subtype
        type = "T cell"
        positivity = [5, 1, 1, 4, 1]
        ann_type = panel.annotation_phenotype(type, positivity)
        self.assertEqual(ann_type.main, type)
        self.assertEqual(ann_type.detailed, "Thelp")

        # Annotation with subtype
        # Annotated expression lower than or equal to the cutoff (default 3) are treated as negative
        type = "T cell"
        positivity = [5, 3, 2, 2, 3]
        ann_type = panel.annotation_phenotype(type, positivity)
        self.assertEqual(ann_type.main, type)
        self.assertEqual(ann_type.detailed, "Thelp")

        type = "T cell"
        positivity = [5, 4, 1, 1, 1]
        ann_type = panel.annotation_phenotype(type, positivity)
        self.assertEqual(ann_type.main, type)
        self.assertEqual(ann_type.detailed, "Treg")

        # Annotation with subtype
        type = "T cell"
        positivity = [1, 5, 1, 4, 1]
        ann_type = panel.annotation_phenotype(type, positivity)
        self.assertEqual(ann_type.main, type)
        self.assertEqual(ann_type.detailed, "Treg")

        # Annotation with subtype
        type = "T cell"
        positivity = [5, 2, 1, 3, 5]
        ann_type = panel.annotation_phenotype(type, positivity)
        self.assertEqual(ann_type.main, type)
        self.assertEqual(ann_type.detailed, "Tcyt")

        # Annotation with subtype
        type = "T cell"
        positivity = [5, 2, 1, 5, 5]
        ann_type = panel.annotation_phenotype(type, positivity)
        self.assertEqual(ann_type.main, type)
        self.assertEqual(ann_type.detailed, "Tcyt")

        # Annotation with subtype
        type = "T cell"
        positivity = [1, 1, 1, 1, 4]
        ann_type = panel.annotation_phenotype(type, positivity)
        self.assertEqual(ann_type.main, type)
        self.assertEqual(ann_type.detailed, "Tcyt")

        # Annotation with subtype
        type = "T cell"
        positivity = [2, 2, 1, 5, 1]
        ann_type = panel.annotation_phenotype(type, positivity)
        self.assertEqual(ann_type.main, type)
        self.assertEqual(ann_type.detailed, "Tmemory")

        # Annotation with subtype
        # Positivity that does not match any phenotype should result in the invalid AnnotationType
        type = "T cell"
        positivity = [5, 4, 1, 1, 5]
        ann_type = panel.annotation_phenotype(type, positivity)
        self.assertTrue(ann_type.invalid)

        type = "T cell"
        positivity = [5, 2, 5, 2, 1]
        ann_type = panel.annotation_phenotype(type, positivity)
        self.assertTrue(ann_type.invalid)

        # Test that positivity cutoff is applied when provided
        # Annotation with subtype
        type = "T cell"
        positivity = [3, 1, 2, 2, 3]
        ann_type = panel.annotation_phenotype(type, positivity, likert_cutoff=2)
        self.assertEqual(ann_type.main, type)
        self.assertEqual(ann_type.detailed, "Tcyt")


    def test_prediction_phenotype(self):
        panel_dicts = get_test_panel_dics()

        panel_dict = panel_dicts[0]
        panel = Panel.from_dict(panel_dict)

        # Check that expected exceptions are risen
        with self.assertRaisesRegex(ValueError, "activ_th has to be a float between 0 and 1"):
            panel.prediction_phenotype([0.8, 0.1, 0.0, 0.7, 0.01], "0")

        with self.assertRaisesRegex(ValueError, "activ_th has to be a float between 0 and 1"):
            panel.prediction_phenotype([0.8, 0.1, 0.0, 0.7, 0.01], -0.1)

        with self.assertRaisesRegex(ValueError, "activ_th has to be a float between 0 and 1"):
            panel.prediction_phenotype([0.8, 0.1, 0.0, 0.7, 0.01], 1.1)

        with self.assertRaisesRegex(ValueError, "predicted_pheno has to be a list of floats with length that matches a number of markers in a panel"):
            panel.prediction_phenotype("Tcyt")

        with self.assertRaisesRegex(ValueError, "predicted_pheno has to be a list of floats with length that matches a number of markers in a panel"):
            panel.prediction_phenotype([])

        with self.assertRaisesRegex(ValueError, "predicted_pheno has to be a list of floats with length that matches a number of markers in a panel"):
            panel.prediction_phenotype([0.8, 0.1])

        with self.assertRaisesRegex(ValueError, "predicted_pheno has to be a list of floats with length that matches a number of markers in a panel"):
            panel.prediction_phenotype(["0.8", "0.1", "0.1", "0.1", "0.1"])

        # No markers are expressed
        predicted_pheno = [0.1] * 5
        ann_type = panel.prediction_phenotype(predicted_pheno)
        self.assertEqual(ann_type.main, "Other cell")
        self.assertEqual(ann_type.detailed, "Other cell")

        # Phenotype without a subtype
        predicted_pheno = [0.1, 0.1, 0.8, 0.1, 0.1]
        ann_type = panel.prediction_phenotype(predicted_pheno)
        self.assertEqual(ann_type.main, "B cell")
        self.assertEqual(ann_type.detailed, "B cell")

        # Phenotype with a subtype
        predicted_pheno = [0.8, 0.1, 0.1, 0.1, 0.39]
        ann_type = panel.prediction_phenotype(predicted_pheno)
        self.assertEqual(ann_type.main, "T cell")
        self.assertEqual(ann_type.detailed, "Thelp")

        # Phenotype with a subtype
        predicted_pheno = [0.8, 0.1, 0.1, 0.9, 0.1]
        ann_type = panel.prediction_phenotype(predicted_pheno)
        self.assertEqual(ann_type.main, "T cell")
        self.assertEqual(ann_type.detailed, "Thelp")

        # Phenotype with a subtype
        predicted_pheno = [0.8, 0.1, 0.1, 0.1, 0.41]
        ann_type = panel.prediction_phenotype(predicted_pheno)
        self.assertEqual(ann_type.main, "T cell")
        self.assertEqual(ann_type.detailed, "Tcyt")

        # Phenotype with a subtype
        predicted_pheno = [0.399, 0.1, 0.1, 0.1, 1.1]
        ann_type = panel.prediction_phenotype(predicted_pheno)
        self.assertEqual(ann_type.main, "T cell")
        self.assertEqual(ann_type.detailed, "Tcyt")

        # Phenotype with a subtype
        predicted_pheno = [0.1, 0.7, 0.1, 0.8, 0.1]
        ann_type = panel.prediction_phenotype(predicted_pheno)
        self.assertEqual(ann_type.main, "T cell")
        self.assertEqual(ann_type.detailed, "Treg")

        # Phenotype with a subtype
        predicted_pheno = [0.9, 0.7, 0.1, 0.8, 0.1]
        ann_type = panel.prediction_phenotype(predicted_pheno)
        self.assertEqual(ann_type.main, "T cell")
        self.assertEqual(ann_type.detailed, "Treg")

        # Phenotype with a subtype
        predicted_pheno = [0.1, 0.1, 0.1, 0.8, 0.1]
        ann_type = panel.prediction_phenotype(predicted_pheno)
        self.assertEqual(ann_type.main, "T cell")
        self.assertEqual(ann_type.detailed, "Tmemory")

        # Invalid phenotype predicted
        predicted_pheno = [0.8, 0.1, 0.7, 0.8, 0.1]
        ann_type = panel.prediction_phenotype(predicted_pheno)
        self.assertTrue(ann_type.invalid)

        predicted_pheno = [0.8, 0.9, 0.39, 0.8, 0.9]
        ann_type = panel.prediction_phenotype(predicted_pheno)
        self.assertTrue(ann_type.invalid)

        # Test that applies a provided threshold
        # Phenotype with a subtype
        predicted_pheno = [0.8, 0.1, 0.19, 0.1, 0.21]
        ann_type = panel.prediction_phenotype(predicted_pheno, 0.2)
        self.assertEqual(ann_type.main, "T cell")
        self.assertEqual(ann_type.detailed, "Tcyt")


if __name__ == '__main__':
    unittest.main()