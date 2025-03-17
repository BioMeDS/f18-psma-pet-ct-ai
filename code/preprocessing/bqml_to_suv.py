import pydicom
import dateutil
import sys

# Taken from https://github.com/qurit/rt-utils/blob/main/NIFTI_conversion.py - under MIT License by Qurit
def bqml_to_suv(dcm_file: pydicom.FileDataset) -> float:
    '''
    Calculates the SUV conversion factor from Bq/mL to g/mL using DICOM header info.
    This simplified version returns only the SUV factor.
    '''
    nuclide_dose = dcm_file[0x054, 0x0016][0][0x0018, 0x1074].value  # Injected dose (Bq)
    weight = dcm_file[0x0010, 0x1030].value  # Patient weight (kg)
    half_life = float(dcm_file[0x054, 0x0016][0][0x0018, 0x1075].value)  # Half life (s)

    series_time = str(dcm_file[0x0008, 0x0031].value)  # Series time (HHMMSS)
    series_date = str(dcm_file[0x0008, 0x0021].value)  # Series date (YYYYMMDD)
    series_dt = dateutil.parser.parse(series_date + ' ' + series_time)

    nuclide_time = str(dcm_file[0x054, 0x0016][0][0x0018, 0x1072].value)  # Injection time
    nuclide_dt = dateutil.parser.parse(series_date + ' ' + nuclide_time)

    delta_time = (series_dt - nuclide_dt).total_seconds()
    decay_correction = 2 ** (-1 * delta_time/half_life)
    suv_factor = (weight * 1000) / (decay_correction * nuclide_dose)
    return suv_factor

if __name__ == "__main__":
    print(bqml_to_suv(pydicom.dcmread(sys.argv[1])))
    