import os, json, settings

new_states = {}
states: dict[str,str] = dict(json.load(open(settings.STATE_ABBR)))
for s in states:
    new_states[s.upper()] = states[s]
    
with open(settings.STATE_ABBR+'.copy.json', 'w') as f:
    json.dump(new_states,f, ensure_ascii=False, indent=4)