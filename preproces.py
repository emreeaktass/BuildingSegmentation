import file_operations as f
import patch_extractor as pe
import splitting_train_test_validation as split
import data_augmentation as da

if __name__ == '__main__':

    print('Preprocesses Started...\n')
    patch_size = 256
    building_path = 'buildings.tiff'
    color_path = 'ODTU_color/color.bip'
    ndsm_path = 'ndsm_file.tiff'

    print('Reading Files Started...\n')
    ndsm_data = f.read(ndsm_path, patch_size, 'height', False, True, True, True)
    color_data = f.read(color_path, patch_size, None, True, False, False, True)
    building_mask = f.read(building_path, patch_size, 'mask', False, False, False, True)
    building_mask[building_mask > 0] = 1
    print('Reading Files Completed...\n')

    print('Patch Extraction Started...\n')
    pe.extract(color_data, ndsm_data, building_mask, patch_size)
    print('Patch Extraction Completed...\n')

    print('Data Splitting Started...\n')
    split.train_test_validation_split_()
    print('Data Splitting Completed...\n')

    print('Data Augmentation Started...\n')
    da.augmentation_()
    print('Data Augmentation Completed...\n')
    print('Preprocesses Completed...\n')