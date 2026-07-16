import os
import shutil

def main():
    src_root = "data/logs"
    target_root = "data/logs/figure2"

    print("Starting configuration logs cleanup...")

    # We want to find all directories like:
    # data/logs/combgen_pentominos_rotation_<shape_name>_seed_<seed>/<model_name>/<timestamp>/
    # and move their contents to:
    # data/logs/figure2/combgen_pentominos_rotation_<shape_name>_seed_<seed>/<model_name>/<timestamp>/

    if not os.path.exists(src_root):
        print(f"Source root {src_root} does not exist. Nothing to clean up.")
        return

    # Find all subdirectories that start with combgen_pentominos_rotation_
    for item in os.listdir(src_root):
        if item == "figure2" or item.startswith("."):
            continue
        
        item_path = os.path.join(src_root, item)
        if not os.path.isdir(item_path):
            continue
            
        if not item.startswith("combgen_pentominos_rotation_"):
            continue
            
        # Walk through model subdirectories (e.g. sa, wae)
        for model_name in os.listdir(item_path):
            model_path = os.path.join(item_path, model_name)
            if not os.path.isdir(model_path):
                continue
                
            # Walk through timestamp subdirectories (e.g. 2026-07-14_15-30)
            for timestamp in os.listdir(model_path):
                timestamp_path = os.path.join(model_path, timestamp)
                if not os.path.isdir(timestamp_path):
                    continue
                    
                # Define target path
                target_path = os.path.join(target_root, item, model_name, timestamp)
                
                # If target path does not exist, we create it
                if not os.path.isdir(target_path):
                    print(f"Target directory {target_path} does not exist. Creating it.")
                    os.makedirs(target_path, exist_ok=True)
                    
                # Move all contents of timestamp_path to target_path
                print(f"Moving contents of {timestamp_path} to {target_path}...")
                for subitem in os.listdir(timestamp_path):
                    subitem_src = os.path.join(timestamp_path, subitem)
                    subitem_target = os.path.join(target_path, subitem)
                    
                    if os.path.exists(subitem_target):
                        if os.path.isdir(subitem_src):
                            shutil.rmtree(subitem_target)
                            shutil.move(subitem_src, subitem_target)
                        else:
                            os.remove(subitem_target)
                            shutil.move(subitem_src, subitem_target)
                    else:
                        shutil.move(subitem_src, subitem_target)
                
                # Remove the now empty timestamp directory
                try:
                    os.rmdir(timestamp_path)
                except OSError as e:
                    print(f"Could not remove {timestamp_path}: {e}")
                    
            # Remove empty model directory
            try:
                os.rmdir(model_path)
            except OSError:
                pass
                
        # Remove empty condition directory
        try:
            os.rmdir(item_path)
        except OSError:
            pass

    print("Cleanup script execution complete!")

if __name__ == "__main__":
    main()
