import json
import os
import pandas as pd
import plotly.express as px
import sys

from datetime import datetime
from typing import Dict, List


def read_file(filename: str) -> Dict:
    with open(filename, "r") as file:
        return json.load(file)


def aggregated(df: pd.DataFrame) -> (pd.DataFrame, int):
    column_names = list(df.columns)

    df = df.pivot(
        index='Time', columns='Process', values=column_names[2]
    ).interpolate(method='time', limit_direction='both')

    result = pd.DataFrame(index=df.index)
    processes = set([p.split('|')[0] for p in list(df.columns)])

    for p in processes:
        df_filtered = df.filter(regex='^' + p)
        result[p + '|sum'] = df_filtered.sum(axis=1)
        result[p + '|mean'] = df_filtered.mean(axis=1)
        result[p + '|max'] = df_filtered.max(axis=1)
        result[p + '|min'] = df_filtered.min(axis=1)

    result['ovn|sum'] = df.filter(regex=r'^ovn.*\|ovn-(central|scale).*').sum(
        axis=1
    )
    ovn_max = result['ovn|sum'].astype('int').max()

    result['ovs|sum'] = df.filter(regex=r'^ovs.*\|ovn-(central|scale).*').sum(
        axis=1
    )

    result = result.astype('int').reset_index().melt(id_vars=['Time'])
    result.columns = column_names

    return result, ovn_max


def resource_stats_generate(
    filename: str, data: Dict, aggregate: bool
) -> None:
    rss: List[List] = []
    cpu: List[List] = []

    for ts, time_slice in sorted(data.items()):
        for name, res in time_slice.items():
            tme = datetime.fromtimestamp(float(ts))
            rss_mb = int(res['rss']) >> 20
            rss.append([tme, name, rss_mb])
            cpu.append([tme, name, float(res['cpu'])])

    df_rss = pd.DataFrame(rss, columns=['Time', 'Process', 'RSS (MB)'])
    df_cpu = pd.DataFrame(cpu, columns=['Time', 'Process', 'CPU (%)'])

    if aggregate:
        df_rss, max_sum_rss = aggregated(df_rss)
        df_cpu, max_sum_cpu = aggregated(df_cpu)

    rss_chart = px.line(
        df_rss,
        x='Time',
        y='RSS (MB)',
        color='Process',
        title=('Aggregate ' if aggregate else '') + 'Resident Set Size',
    )
    cpu_chart = px.line(
        df_cpu,
        x='Time',
        y='CPU (%)',
        color='Process',
        title=('Aggregate ' if aggregate else '') + 'CPU usage',
    )

    with open(filename, 'w') as report_file:
        report_file.write('<html>')
        if aggregate:
            report_file.write(
                '''
                <table border="1" class="dataframe">
                <tbody>
                    <tr>
                        <td>Max(Sum(OVN RSS))</td>
                        <td>'''
                + str(max_sum_rss)
                + ''' MB</td>
                    </tr>
                    <tr>
                        <td>Max(Sum(OVN CPU))</td>
                        <td>'''
                + str(max_sum_cpu)
                + ''' %</td>
                    </tr>
                </tbody>
                </table>
            '''
            )
        report_file.write(
            rss_chart.to_html(
                full_html=False,
                include_plotlyjs='cdn',
                default_width='90%',
                default_height='90%',
            )
        )
        report_file.write(
            cpu_chart.to_html(
                full_html=False,
                include_plotlyjs='cdn',
                default_width='90%',
                default_height='90%',
            )
        )
        report_file.write('</html>')


if __name__ == '__main__':
    if len(sys.argv) < 3 or (
        sys.argv[1] == '--aggregate' and len(sys.argv) < 4
    ):
        print(
            f'Usage: {sys.argv[0]} '
            + '[--aggregate] output-file input-file [input-file ...]'
        )
        sys.exit(1)

    aggregate = sys.argv[1] == '--aggregate'
    start = 2 if aggregate else 1

    if os.path.isfile(sys.argv[start]):
        print(f'Output file {sys.argv[start]} already exists')
        sys.exit(2)

    print(f'Processing stats from {len(sys.argv) - start - 1} files.')

    data: Dict = {}
    for f in sys.argv[start + 1 :]:
        d = read_file(f)
        data.update(d)

    resource_stats_generate(sys.argv[start], data, aggregate)
