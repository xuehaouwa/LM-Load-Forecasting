import os
import datetime
import inflect

import pandas as pd


# "the electric load recorded at 02:00:00 on January 1st, 2021 with a value of 3.6288895805555557 kW"

def time_transform(time_data):
    p = inflect.engine()
    date = datetime.datetime.strptime(time_data, '%Y-%m-%d %H:%M:%S')
    # out1 = date.strftime('%p %B %d, %Y, %A, %I')
    out1 = date.strftime('%A %I %p')
    out2 = date.strftime('%B %-d')
    i = date.strftime('%-d')
    sti = p.ordinal(int(i))
    year = date.strftime('%Y')
    out2 = out2.replace(i, sti)
    return out1, out2, year


def caption(time_data, load_data):
    time_1, time_2, time_3 = time_transform(time_data)

    return f"The electric load recorded at {time_1} on {time_2}, {time_3} was {load_data} kW."


def generate_captions(csv_path, save_file):
    raw_df = pd.read_csv(csv_path)
    output = []
    for i in range(len(raw_df)):
        load = float(raw_df['Total_Load'][i])
        load = round(load, 2)
        time_step = str((raw_df['Timestamp'][i]))
        prompt = caption(time_step, load)
        output.append(prompt)

    with open(save_file, "w") as f:
        for i in output:
            f.write(i + "\n")
        f.close()


if __name__ == "__main__":

    generate_captions("building_A.csv",
                      "building_A.txt")

