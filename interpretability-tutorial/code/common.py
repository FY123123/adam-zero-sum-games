"""Model loading and the indirect-object prompts shared by the GPT-2 experiments."""
import torch
from transformer_lens import TransformerBridge


def load_model():
    """GPT-2 small. Compatibility mode folds every LayerNorm into x -> (x - mean(x)) / scale(x)."""
    model = TransformerBridge.boot_transformers("gpt2")
    model.enable_compatibility_mode()
    return model


IOI_PAIRS = [(" Mary", " John"), (" Tom", " James"), (" Dan", " Sarah"),
             (" Anna", " Mark"), (" Kate", " Paul"), (" Lisa", " David")]
IOI_PAIRS = IOI_PAIRS + [(b, a) for a, b in IOI_PAIRS]
IOI_TEMPLATE = "When{A} and{B} went to the store,{S} gave a drink to"
IOI_POSITIONS = ["BOS", "When", "IO", "and", "S1", "went", "to", "the", "store", ",", "S2", "gave", "a", "drink", "to"]


def ioi_prompts():
    """Clean prompts end in the repeated name B (answer A). Corrupted prompts repeat A (answer B)."""
    clean = [IOI_TEMPLATE.format(A=a, B=b, S=b) for a, b in IOI_PAIRS]
    corrupt = [IOI_TEMPLATE.format(A=a, B=b, S=a) for a, b in IOI_PAIRS]
    return clean, corrupt


# Single-token first names for the experiments on the gender of a name (sessions 3 and 5).
MALE = "John James Robert Michael William David Richard Joseph Thomas Charles Daniel Matthew Anthony Mark Paul Steven Andrew Kevin Brian George Edward Peter Jack Henry Frank Tom Bob Jim Steve Mike Adam Ben Luke Ryan Jason Eric Scott Simon Martin Alan Carl Fred Harry Jake Tim Tony Walter Roger Oliver Patrick Sean Kyle Justin Aaron Arthur Bruce".split()
FEMALE = "Mary Sarah Jennifer Linda Elizabeth Susan Jessica Karen Nancy Lisa Margaret Betty Sandra Ashley Emily Anna Laura Rachel Alice Kate Emma Julia Maria Helen Amy Jane Lucy Grace Rose Sophie Claire Diana Ellen Hannah Jenny Katie Lily Michelle Nicole Olivia Rebecca Sara Victoria Amanda Angela Barbara Carol Catherine Christina Donna Heather Joan Julie Kelly Megan Melissa Molly Natalie Ruth Samantha Stephanie".split()[:len(MALE)]
GENDER_TEMPLATE, NAME_POS = "After {} got home,", 2
