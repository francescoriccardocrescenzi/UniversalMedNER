# REWARD FUNCTION PSEUDOCODE

# INPUTS
TEXT = str
ENT_TYPES = List[str]

# TARGET (keys(ENTITIES) = ENT_TYPES)
ENTITIES = Dict[str -> List[str]]

# OUTPUT
OUT_TEXT = str

R = 0

# --- LEVEL 1: JSON PARSING ---
# Reward valid JSON with correct structure (Dict[str -> List[str]]).
# Returns 0 for anything unparsable or structurally wrong.

if not isParsable(OUT_TEXT):
  return R
OUT_DICT = parseJson(OUT_TEXT)
if not hasCorrectStructure(OUT_DICT):  # must be Dict[str -> List[str]]
  return R
R += 1

# --- LEVEL 2: KEY MATCHING ---
# Reward the model for outputting the correct entity type keys.
# Normalize by N_TYPES (expected count) so that missing keys reduce the reward.

N_TYPES = len(ENT_TYPES)

for KEY in keys(OUT_DICT):
  if KEY not in ENT_TYPES:
    return R          # wrong key found: stop here, no bonus, no quality component
  R += 1 / N_TYPES   # partial credit per correct key, normalized by expected count

# Check for missing keys: all output keys passed the check above, so if the
# counts differ the model omitted some expected types. Cap reward here.
if len(OUT_DICT) != N_TYPES:
  return R

# bonus buffer (B = 1 recommended)
# needed as margin for the possible negative signal from the quality component, while avoiding overlap
R += B

# --- LEVEL 3: EXTRACTION QUALITY ---
# Compute a per-type score for each expected entity type.
# For negative types (empty target list):
#   - empty output list -> score = +1
#   - non-empty output list   -> score = -1
# For positive types, use computeExtractionScore (see below).
# Normalize each per-type score by N_TYPES for cross-sample comparability.
# Clip Q_R at -B to preserve reward ordering (correct format always beats wrong format).

Q_R = 0

for T in ENT_TYPES:
  T_LIST = ENTITIES[T]
  O_LIST = OUT_DICT[T]           # safe: exact key match enforced above

  if len(T_LIST) == 0: # first handle the negative type case
    if len(O_LIST) == 0:
      SCORE = 1                  # correct negative: model correctly output []
    else:
      SCORE = -1                 # hallucination on a negative type
  else:
    SCORE = computeExtractionScore(T_LIST, O_LIST)

  Q_R += SCORE / N_TYPES

# Clip to prevent quality penalty from erasing the format reward.
# B + Q_R is the net impact on R from this point; keeping it >= 0 ensures
# any output that passed the key check still scores higher than wrong format (R=0).
if B + Q_R < 0:
  Q_R = -B

R += Q_R

# --- EXTRACTION SCORE ---
# Represents extraction quality for a single positive entity type.
# Rewards:
#   +1/N for each GT span correctly predicted (exact match, no duplicates)
# Penalizes:
#   -1/N for each wrong prediction or duplicate of a correct prediction  (precision signal)
#   -1/N for each GT span that was never predicted                        (recall signal)
# The recall penalty (missed entities) was added to ensure the model has an
# incentive to extract ALL entities, not just avoid hallucinations.
# Without it, predicting nothing gives SCORE = 0 which beats any attempt
# that risks a wrong prediction.

def computeExtractionScore(T_LIST: List[str], O_LIST: List[str]) -> float:
  N = len(T_LIST)
  SCORE = 0
  MATCHES = []          # tracks which predicted spans have been credited (by value)

  for OE in O_LIST:
    if OE in T_LIST and OE not in MATCHES:
      MATCHES.append(OE)
      SCORE += 1 / N    # correct extraction (precision + recall signal)
    else:
      SCORE -= 1 / N    # wrong prediction or duplicate of already-matched span

  # Recall signal: penalize GT spans that were never matched
  N_MISSED = N - len(MATCHES)
  SCORE -= N_MISSED / N

  return SCORE     # normalized [-inf, 1], there is no lower bound since the model can (theoretically) generate infinite wrong entities, but only N correct ones

  # NOTE: all checks like "OE in T_LIST" or "OE not in MATCHES" are ALL performed with exact match, to ensure the model is rewarded for perfect grounding (which should not be hard if the target extracted labels appear exactly the same in text)
  # Either using case sentitive or insensitive matching is viable, depending on what we need.
  # If the most popular NER metrics are case insensitive, then we can go with that to simplify the task 