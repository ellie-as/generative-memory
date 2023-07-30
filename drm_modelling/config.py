# remove lists for king, girl, man, black
DRM_data = {
    "anger": ["fear", "temper", "hatred", "fury", "happy", "enrage", "emotion", "rage", "hate", "mean", "ire", "mad", "wrath", "calm", "fight"],
    "thief": ["steal", "robber", "jail", "villian", "bandit", "criminal", "rob", "cop", "money", "bad", "burglar", "crook", "crime", "gun", "bank"],
    "fruit": ["ripe", "citrus", "vegetable", "juice", "cocktail", "banana", "orange", "basket", "bowl", "salad", "berry", "kiwi", "pear", "apple", "cherry"],
    "high": ["sky", "dive", "building", "up", "low", "over", "above", "tall", "noon", "airplane", "elevate", "jump", "tower", "clouds", "cliff"],
    "window": ["sill", "shade", "screen", "ledge", "sash", "door", "view", "house", "glass", "shutter", "frame", "breeze", "curtain", "pane", "open"],
    "slow": ["delay", "lethargic", "molasses", "sluggish", "traffic", "wait", "hesitant", "speed", "fast", "listless", "stop", "snail", "quick", "turtle", "cautious"],
    "foot": ["hand", "smell", "toe", "walk", "kick", "ankle", "inch", "mouth", "sandals", "arm", "yard", "sock", "boot", "soccer", "shoe"],
    "chair": ["desk", "cushion", "couch", "bench", "sit", "swivel", "sofa", "recliner", "rocking", "sitting", "legs", "table", "seat", "wood", "stool"],
    # "black": ["color", "charred", "gray", "bottom", "ink", "night", "coal", "dark", "cat", "blue", "funeral", "grief", "death", "white", "brown"],
    "sleep": ["rest", "bed", "nap", "peace", "drowsy", "blanket", "doze", "tired", "awake", "snooze", "yawn", "slumber", "snore", "wake", "dream"],
    "music": ["jazz", "horn", "concert", "orchestra", "rhythm", "sing", "piano", "band", "note", "instrument", "art", "sound", "symphony", "radio", "melody"],
    # "man": ["beard", "lady", "friend", "woman", "male", "husband", "strong", "handsome", "suit", "muscle", "old", "uncle", "mouse", "person", "father"],
    "bread": ["flour", "toast", "jam", "milk", "rye", "loaf", "sandwich", "eat", "butter", "slice", "dough", "crust", "food", "wine", "jelly"],
    "sweet": ["soda", "heart", "tooth", "tart", "taste", "sour", "bitter", "good", "sugar", "candy", "nice", "cake", "pie", "chocolate", "honey"],
    "spider": ["fly", "insect", "animal", "ugly", "tarantula", "poison", "bug", "bite", "feelers", "creepy", "web", "arachnid", "small", "fright", "crawl"],
    "soft": ["loud", "downy", "furry", "cotton", "touch", "light", "feather", "skin", "tender", "plush", "pillow", "fur", "fluffy", "hard", "kitten"],
    "needle": ["thread", "point", "cloth", "sharp", "pin", "eye", "hurt", "knitting", "sewing", "injection", "syringe", "prick", "thorn", "thimble", "haystack"],
    # "king": ["palace", "crown", "dictator", "throne", "chess", "george", "queen", "rule", "leader", "subjects", "monarch", "royal", "reign", "prince", "england"],
    "mountain": ["climber", "valley", "summit", "plain", "hill", "bike", "peak", "ski", "molehill", "goat", "glacier", "steep", "climb", "range", "top"],
    "doctor": ["lawyer", "clinic", "health", "medicine", "sick", "stethoscope", "cure", "nurse", "surgeon", "patient", "hospital", "dentist", "physician", "ill", "office"],
    "cold": ["chilly", "hot", "wet", "winter", "freeze", "frigid", "heat", "snow", "arctic", "air", "weather", "shiver", "ice", "frost", "warm"],
    # "girl": ["sister", "daughter", "pretty", "dance", "boy", "female", "niece", "young", "date", "aunt", "cute", "dolls", "hair", "beautiful", "dress"],
    "river": ["barge", "mississippi", "brook", "creek", "swim", "stream", "bridge", "flow", "fish", "water", "run", "tide", "lake", "winding", "boat"],
    "rough": ["tough", "smooth", "ground", "uneven", "sand", "bumpy", "rugged", "boards", "road", "sandpaper", "ready", "riders", "gravel", "jagged", "coarse"]
}

lures = list(DRM_data.keys())
DRM_lists = list(DRM_data.values())
