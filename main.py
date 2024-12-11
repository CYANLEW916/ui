import streamlit as st
import struct
import csv
from datetime import datetime
import io
import pandas as pd
import re
import string
# Constants and mappings
START_SEGMENT = 1
LINE_SEGMENT = 2
ARC_SEGMENT = 3

SEGMENT_TYPES = {
    START_SEGMENT: "Start Segment",
    LINE_SEGMENT: "Line Segment",
    ARC_SEGMENT: "Arc Segment"
}

CHARACTERISTICS_MAP = {
    1: "Start of Climb",
    2: "Top of Climb",
    3: "Top of Descent",
    6: "Runway",
    9: "Aircraft is currently flying",
    10: "Discontinuity",
    11: "Non-Flyable",
    12: "Level Off"
}

# Helper functions
def binary_to_int(binary):
    return int(binary, 2)


def hex_to_binary(hex_string):
    hex_string = ''.join(hex_string.split())
    try:
        binary = ''.join(format(int(char, 16), '04b') for char in hex_string)
    except ValueError:
        raise ValueError("Hexadecimal data contains invalid characters.")
    return binary


def binary_to_signed_int(binary, bit_length):
    value = int(binary, 2)
    if value >= 2 ** (bit_length - 1):
        value -= 2 ** bit_length
    return value


def binary_to_float(binary):
    return struct.unpack('!f', struct.pack('!I', int(binary, 2)))[0]


def parse_time(binary):
    if len(binary) < 32:
        raise ValueError("Time binary data is incomplete.")
    milliseconds = binary_to_int(binary[0:11])  # Bits 0-10
    seconds = binary_to_int(binary[11:18])  # Bits 11-17
    minutes = binary_to_int(binary[18:25])  # Bits 18-24
    hours = binary_to_int(binary[25:32])  # Bits 25-31
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def parse_characteristics(binary):
    characteristics_value = binary_to_int(binary)
    if characteristics_value == 0:
        return "0 = Normal segment"
    descriptions = []
    for bit, description in CHARACTERISTICS_MAP.items():
        if characteristics_value & (1 << (24 - bit)):
            descriptions.append(f"{bit} = {description}")
    return ', '.join(descriptions) if descriptions else "No characteristics set"


def parse_segment(segment):
    if len(segment) < 128:
        raise ValueError("Segment data is incomplete.")

    segment_type = binary_to_int(segment[:3])
    data_type = binary_to_int(segment[3:8])
    characteristics_binary = segment[8:32]
    path_rnp_binary = segment[32:64]
    path_rnp_hex = hex(int(path_rnp_binary, 2))[2:].zfill(8).upper()
    path_rnp_value = binary_to_float(path_rnp_binary)
    latitude_binary = segment[64:96]
    latitude_value = binary_to_float(latitude_binary)
    longitude_binary = segment[96:128]
    longitude_value = binary_to_float(longitude_binary)

    parsed_data = [
        ("Geometry", segment[:3], SEGMENT_TYPES.get(segment_type, "Unknown")),
        ("Data type", segment[3:8], f"{data_type} = supporting the ETA"),
        ("Characteristics", characteristics_binary, parse_characteristics(characteristics_binary)),
        ("Path RNP", path_rnp_binary,
         f"{path_rnp_value:.7f} NM" if path_rnp_hex != "FF800000" else "hxFF800000 = invalid"),
        ("Point Latitude", latitude_binary, f"{latitude_value:.7f}"),
        ("Point Longitude", longitude_binary, f"{longitude_value:.8f}")
    ]

    if segment_type in [START_SEGMENT, LINE_SEGMENT]:
        if len(segment) < 192:
            raise ValueError("Segment data is incomplete for START_SEGMENT or LINE_SEGMENT.")
        altitude_binary = segment[128:160]
        altitude_value = binary_to_signed_int(altitude_binary, 32)
        eta_binary = segment[160:192]
        parsed_data.extend([
            ("Point Altitude", altitude_binary,
             f"{altitude_value} ft" if altitude_binary != "11111111100000000000000000000000" else "hxFF800000 = invalid"),
            ("Point ETA", eta_binary, parse_time(eta_binary))
        ])
    elif segment_type == ARC_SEGMENT:
        if len(segment) < 288:
            raise ValueError("Segment data is incomplete for ARC_SEGMENT.")
        turn_radius_binary = segment[128:160]
        turn_radius_value = binary_to_float(turn_radius_binary)
        turn_center_lat_binary = segment[160:192]
        turn_center_lat_value = binary_to_float(turn_center_lat_binary)
        turn_center_lon_binary = segment[192:224]
        turn_center_lon_value = binary_to_float(turn_center_lon_binary)
        altitude_binary = segment[224:256]
        altitude_value = binary_to_signed_int(altitude_binary, 32)
        eta_binary = segment[256:288]
        parsed_data.extend([
            ("Turn Radius", turn_radius_binary, f"{turn_radius_value:.7f} NM"),
            ("Turn Center Latitude", turn_center_lat_binary, f"{turn_center_lat_value:.7f}"),
            ("Turn Center Longitude", turn_center_lon_binary, f"{turn_center_lon_value:.8f}"),
            ("Point Altitude", altitude_binary,
             f"{altitude_value} ft" if altitude_binary != "11111111100000000000000000000000" else "hxFF800000 = invalid"),
            ("Point ETA", eta_binary, parse_time(eta_binary))
        ])

    return SEGMENT_TYPES.get(segment_type, "Unknown Segment"), parsed_data


def parse_cell_range(cell_range):
    """
    Parses a cell range string (e.g., 'B2:B33') and returns the column letter and start/end rows.
    """
    match = re.match(r'^([A-Za-z]+)(\d+):([A-Za-z]+)(\d+)$', cell_range.strip())
    if not match:
        raise ValueError("Invalid cell range format. Use format like 'B2:B33'.")

    start_col, start_row, end_col, end_row = match.groups()

    if start_col.upper() != end_col.upper():
        raise ValueError("Start and end columns must be the same.")

    return start_col.upper(), int(start_row), int(end_row)


def process_flight_plan(hex_data):
    binary_data = hex_to_binary(hex_data)
    segments = []
    i = 0
    while i < len(binary_data):
        if i + 3 > len(binary_data):
            st.warning(f"Incomplete segment type bits at bit index {i}. Skipping remaining bits.")
            break  # Avoid slicing beyond the string
        segment_type = binary_to_int(binary_data[i:i + 3])
        if segment_type in [START_SEGMENT, LINE_SEGMENT]:
            segment_length = 192
        elif segment_type == ARC_SEGMENT:
            segment_length = 288
        else:
            st.warning(f"Unknown segment type {segment_type} at bit index {i}. Skipping 3 bits.")
            i += 3  # Move past the segment type bits
            continue
        segment = binary_data[i:i + segment_length]
        if len(segment) < segment_length:
            st.warning(
                f"Incomplete segment at bit index {i}. Expected {segment_length} bits, got {len(segment)} bits. Skipping.")
            break
        try:
            parsed_segment = parse_segment(segment)
            segments.append(parsed_segment)
        except ValueError as ve:
            st.warning(f"Error parsing segment at bit index {i}: {ve}. Skipping segment.")
        i += segment_length

    if not segments:
        raise ValueError("No valid segments were parsed.")

    # Prepare CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["BLOCK DESCRIPTION", "DATA", "Source Data", "Engineering Value"])
    for segment_type, segment_data in segments:
        writer.writerow([segment_type, "", "", ""])
        for field in segment_data:
            writer.writerow([""] + list(field))

    return output.getvalue()

# Helper function to convert Excel column letters to zero-based indices
def excel_column_to_index(column_letter):
    """
    Convert Excel column letter to zero-based index.
    Example: 'A' -> 0, 'B' -> 1, ..., 'AA' -> 26, etc.
    """
    column_letter = column_letter.upper()
    expn = 0
    col_index = 0
    for char in reversed(column_letter):
        col_index += (string.ascii_uppercase.index(char) + 1) * (26 ** expn)
        expn += 1
    return col_index - 1

# Placeholder functions
def is_valid_hex(hex_str):
    """Check if the string is a valid hexadecimal."""
    try:
        int(hex_str, 16)
        return True
    except ValueError:
        return False


# Streamlit App
def main():
    st.title("Flight Plan Processor")
    st.write("""
        **Flight Plan Processor**

        Upload an Excel file containing hexadecimal flight plan data, select the column to process, specify the row range, and generate a CSV file with parsed data.

        **Instructions:**
        1. **Upload Excel File:** Click on the "Upload Excel File" button and select your `.xlsx` or `.xls` file.
        2. **Select Column:** Choose the column (by its Excel letter, e.g., A, B, C) that contains the hexadecimal data.
        3. **Specify Row Range:** Enter the start and end rows (1-based indexing) you wish to process.
        4. **Process Data:** Click the "Process" button to generate and download the parsed CSV file.
    """)

    # File uploader for Excel files
    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls"])

    if uploaded_file is not None:
        try:
            # Read the Excel file without headers
            df = pd.read_excel(uploaded_file, engine='openpyxl', header=None)
            st.success("Excel file uploaded successfully!")

            # Display the DataFrame preview
            st.write("### Preview of Uploaded Excel Data")
            st.dataframe(df.head())

            # Generate Excel-style column labels (A, B, C, ...)
            num_columns = df.shape[1]
            column_labels = []
            for i in range(num_columns):
                column_label = ""
                n = i + 1
                while n > 0:
                    n, remainder = divmod(n - 1, 26)
                    column_label = chr(65 + remainder) + column_label
                column_labels.append(column_label)

            st.write(f"**Total Columns in Excel File: {num_columns} ({', '.join(column_labels)})**")

            # Column selection
            selected_column = st.selectbox(
                "Select Column Containing Hexadecimal Data (e.g., A, B, C)",
                options=column_labels,
                format_func=lambda x: x  # Display as-is (A, B, C, ...)
            )

            # Convert column letter to zero-based index
            selected_col_idx = excel_column_to_index(selected_column)

            # Get total number of rows
            total_rows = len(df)
            st.write(f"**Total Rows in Excel File: {total_rows}**")

            # Input for row range (1-based indexing)
            st.subheader("Specify Row Range to Process")
            start_row = st.number_input(
                "Start Row",
                min_value=1,
                max_value=total_rows,
                step=1,
                value=1
            )
            end_row = st.number_input(
                "End Row",
                min_value=start_row,
                max_value=total_rows,
                step=1,
                value=total_rows
            )

            if st.button("Process"):
                try:
                    # Convert to 0-based indexing for pandas
                    data_start_idx = int(start_row) - 1
                    data_end_idx = int(end_row)  # iloc is exclusive at the end

                    # Extract data from the specified range
                    data_series = df.iloc[data_start_idx:data_end_idx, selected_col_idx].dropna().astype(str).tolist()
                    original_rows = df.iloc[data_start_idx:data_end_idx,
                                    selected_col_idx].dropna().index + 1  # Original Excel row numbers

                    if not data_series:
                        st.error("No data found in the specified range.")
                        return

                    st.write(
                        f"### Processing {len(data_series)} hexadecimal entries from column '{selected_column}', rows {start_row} to {end_row}."
                    )

                    # Initialize in-memory CSV for aggregated results
                    all_parsed_csv = io.StringIO()
                    writer = csv.writer(all_parsed_csv)
                    # Including Original Excel Row Number
                    writer.writerow(["Entry Number", "Excel Row Number", "BLOCK DESCRIPTION", "DATA", "Source Data",
                                     "Engineering Value"])

                    errors = []

                    progress_bar = st.progress(0)
                    for idx, (hex_data, excel_row) in enumerate(zip(data_series, original_rows), start=1):
                        # Validate hexadecimal data
                        if not is_valid_hex(hex_data):
                            errors.append(
                                f"Entry {idx} (Excel Row {excel_row}) contains invalid hexadecimal characters. Skipping.")
                            progress_bar.progress(idx / len(data_series))
                            continue
                        try:
                            csv_content = process_flight_plan(hex_data)
                            csv_reader = csv.reader(io.StringIO(csv_content))
                            # Skip header in the processed content
                            next(csv_reader, None)
                            for row in csv_reader:
                                # Prepend the entry number and original Excel row number
                                writer.writerow([idx, excel_row] + row)
                        except Exception as e:
                            errors.append(f"Entry {idx} (Excel Row {excel_row}): {e}")
                        progress_bar.progress(idx / len(data_series))
                    progress_bar.empty()

                    # Display any errors encountered during processing
                    if errors:
                        st.warning("Some entries could not be processed:")
                        for error in errors:
                            st.write(f"- {error}")

                    if all_parsed_csv.tell() == 0:
                        st.error("No valid hexadecimal data was processed.")
                        return

                    st.success("All hexadecimal data has been processed successfully!")

                    # Reset the buffer's position to the beginning
                    all_parsed_csv.seek(0)

                    # Provide a download button for the aggregated CSV
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_file = f"flight_plan_batch_{timestamp}.csv"

                    st.download_button(
                        label="Download Parsed CSV",
                        data=all_parsed_csv.getvalue(),
                        file_name=output_file,
                        mime='text/csv',
                    )

                except Exception as e:
                    st.error(f"An error occurred while processing the data: {e}")

        except Exception as e:
            st.error(f"Failed to read the Excel file: {e}")
    else:
        st.info("Please upload an Excel file to begin.")


if __name__ == "__main__":
    main()