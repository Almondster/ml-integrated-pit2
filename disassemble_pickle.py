import pickletools
import collections

modules_classes = collections.defaultdict(set)

print("Disassembling pickle file...")
try:
    with open("major_category_model_for_task2.pkl", "rb") as f:
        for opcode, arg, pos in pickletools.genops(f):
            if opcode.name in ('GLOBAL', 'INST'):
                # arg is a string like "module_name class_name"
                parts = arg.split()
                if len(parts) == 2:
                    modules_classes[parts[0]].add(parts[1])
                else:
                    print(f"Strange global: {arg}")
except Exception as e:
    print(f"Error during disassembly: {e}")

print("\nReferenced Modules and Classes:")
for mod in sorted(modules_classes.keys()):
    print(f"Module: {mod}")
    for cls in sorted(modules_classes[mod]):
        print(f"  - {cls}")
