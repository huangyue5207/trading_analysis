"""
币安K线数据分析和可视化脚本

功能：
1. 将任意时间粒度的K线拆分到1秒K线，计算每根1秒K线的典型价格(high+low+close)/3 * 该1秒K线的成交量，
   然后对所有1秒K线的计算结果求和，最后除以当前粒度K线的总成交量
2. 在K线图中绘制该结果曲线
3. 计算每根K线的(open+close)/2并在K线图中绘制
4. 使用币安API获取数据
"""

import requests
import pandas as pd
import numpy as np
import importlib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import time
import os
import json

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class BinanceKlineData:
    """币安K线数据获取类"""
    
    def __init__(self, csv_file_path: str):
        self.base_url = "https://api.binance.com/api/v3"
        self.session = requests.Session()
        self.csv_file_path = csv_file_path
        
        # 币安时间间隔映射到秒数
        self.interval_seconds = {
            '1s': 1,
            '1m': 60,
            '3m': 180,
            '5m': 300,
            '15m': 900,
            '30m': 1800,
            '1h': 3600,
            '2h': 7200,
            '4h': 14400,
            '6h': 21600,
            '8h': 28800,
            '12h': 43200,
            '1d': 86400,
            '3d': 259200,
            '1w': 604800,
            '1M': 2592000
        }

        self.history_periods = {
            '15m': 3,
            '1h': 12,
            '4h': 48,
            '1d': 192
        }
    
    def get_kline_data(self,
                      symbol: str = "BTCUSDT",
                      interval: str = "1h",
                      start_time: Optional[datetime] = None,
                      end_time: Optional[datetime] = None,
                      limit: int = 1000) -> pd.DataFrame:
        """
        获取币安K线数据
        
        Args:
            symbol: 交易对，如 'BTCUSDT', 'ETHUSDT'
            interval: 时间间隔，如 '1m', '5m', '1h', '1d'
            start_time: 开始时间 (datetime对象)
            end_time: 结束时间 (datetime对象)
            limit: 单次请求最大数量（默认1000）
            
        Returns:
            pandas.DataFrame: 包含OHLCV数据的DataFrame
        """
        
        url = f"{self.base_url}/klines"
        all_data = []
        
        try:
            # 如果没有指定时间，默认获取最近的数据
            if not start_time:
                start_time = datetime.now() - timedelta(days=7)
            if not end_time:
                end_time = datetime.now()
            
            # 转换为时间戳（毫秒）
            start_ms = int(start_time.timestamp() * 1000)
            end_ms = int(end_time.timestamp() * 1000)
            current_start = start_ms
            
            # 分批获取数据（币安API限制单次最多1000条）
            while current_start < end_ms:
                params = {
                    'symbol': symbol,
                    'interval': interval,
                    'startTime': current_start,
                    'endTime': end_ms,
                    'limit': limit
                }
                
                response = self.session.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if not data:
                    break
                
                all_data.extend(data)
                
                # 更新起始时间（使用最后一条数据的时间）
                last_time = data[-1][0]  # 第一条是开盘时间戳
                if last_time >= end_ms:
                    break
                
                current_start = last_time + 1
                
                # 防止API限速
                time.sleep(0.1)
            
            if not all_data:
                print(f"未获取到 {symbol} 的数据")
                return pd.DataFrame()
            
            # 币安返回的数据格式: 
            # [Open time, Open, High, Low, Close, Volume, Close time, Quote volume, 
            #  Trades, Taker buy base volume, Taker buy quote volume, Ignore]
            df = pd.DataFrame(all_data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base_volume',
                'taker_buy_quote_volume', 'ignore'
            ])
            
            # 转换时间戳为datetime
            df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

            # 将df['timestamp']转成utc+8时区
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert('UTC+08:00')
            
            # 转换数值类型
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 选择需要的列
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume']]
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            print(f"成功获取 {symbol} 的 {len(df)} 条K线数据")
            print(f"时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
            
            return df
            
        except requests.RequestException as e:
            print(f"获取K线数据失败: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"处理数据时出错: {e}")
            return pd.DataFrame()
    
    def calculate_typical_price_volume(self, df: pd.DataFrame) -> float:
        """
        将K线拆分到1秒级别，计算典型价格*成交量并汇总，最后除以总成交量
        
        Args:
            df: 包含OHLCV数据的DataFrame
            
        Returns:
            float: 正向与负向典型价格成交量差（净流入）
        """
        positive_price_volume = 0
        negative_price_volume = 0
        
        for idx, row in df.iterrows():
            # 获取当前K线的时间范围
            start_time = row['timestamp']
            
            # 确定时间粒度（秒）
            if idx < len(df) - 1:
                next_time = df.iloc[idx + 1]['timestamp']
                interval_seconds = int((next_time - start_time).total_seconds())
            else:
                # 如果是最后一条，使用前一条的间隔
                if idx > 0:
                    prev_time = df.iloc[idx - 1]['timestamp']
                    interval_seconds = int((start_time - prev_time).total_seconds())
                else:
                    interval_seconds = 3600  # 默认1小时
            
            # 计算典型价格
            typical_price = (row['high'] + row['low'] + row['close']) / 3
            
            # 当前K线的总成交量
            total_volume = row['volume']
            
            # 将当前K线的成交量均匀分配到1秒K线
            # 假设成交量在时间上是均匀分布的
            volume_per_second = total_volume / interval_seconds if interval_seconds > 0 else 0
            
            # 计算每根1秒K线的典型价格*成交量
            # 注意：这里我们假设典型价格在整个K线期间保持不变
            # 实际情况下，1秒内的价格会在high和low之间变化
            # 但为了简化，我们使用典型价格作为平均值
            typical_price_volume_per_second = typical_price * volume_per_second

            # if row['close'] >= row['open']:
            #     positive_price_volume += typical_price_volume_per_second
            # else:
            #     negative_price_volume += typical_price_volume_per_second
            
            if row['close'] > row['open']:
                positive_price_volume += typical_price_volume_per_second
            elif row['close'] < row['open']:
                negative_price_volume += typical_price_volume_per_second
            else:
                if typical_price <= row['close']:
                    positive_price_volume += typical_price_volume_per_second
                else:
                    negative_price_volume += typical_price_volume_per_second
        print(f"positive_price_volume: {positive_price_volume}, negative_price_volume: {negative_price_volume}")
        return (positive_price_volume - negative_price_volume) 
    
    def calculate_mfi(self, positive_price_volume: float, negative_price_volume: float) -> float:
        """
        计算MFI
        
        Args:
            positive_price_volume: 正向价格成交量和
            negative_price_volume: 负向价格成交量和
        """
        return 100 - (100 / (1 + positive_price_volume / negative_price_volume))
    
    def calculate_net_inflow_obv(self, df: pd.DataFrame) -> pd.Series:
        """
        计算基于net_inflow的OBV类似指标（累积净流入指标）
        
        Args:
            df: 包含net_inflow列的DataFrame
            
        Returns:
            pd.Series: 累积净流入指标值
        """
        if 'net_inflow' not in df.columns:
            return pd.Series(dtype=float)
        
        # 累积net_inflow值，类似OBV的累积方式
        obv_values = []
        cumulative_value = 0.0
        
        for net_inflow in df['net_inflow']:
            if pd.notna(net_inflow):
                cumulative_value += net_inflow
            obv_values.append(cumulative_value)
        
        return pd.Series(obv_values, index=df.index)
    
    def plot_with_mplfinance(self, df: pd.DataFrame, symbol: str, interval: str):
        """
        使用 mplfinance 绘制更美观的K线图及MFI指标，并保存为图片文件
        """
        try:
            mpf = importlib.import_module("mplfinance")
        except ImportError as exc:
            raise ImportError("请先安装 mplfinance，执行：pip install mplfinance") from exc

        if df.empty:
            print("数据为空，无法绘制图表")
            return

        # 主面板只包含K线数据，不包含volume
        plot_df = df[['timestamp', 'open', 'high', 'low', 'close']].copy()
        plot_df = plot_df.set_index('timestamp')
        plot_df.index.name = 'Date'
        plot_df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close'
        }, inplace=True)

        add_plots = []
        
        # 添加net_inflow到panel 1（使用柱状图，左侧y轴）
        # 正数绘制绿色，负数绘制红色
        if 'net_inflow' in df.columns:
            net_inflow_series = df.set_index('timestamp')['net_inflow']
            
            # 将net_inflow分成正数和负数两部分
            positive_inflow = net_inflow_series.copy()
            positive_inflow[positive_inflow < 0] = 0  # 只保留正数
            
            negative_inflow = net_inflow_series.copy()
            negative_inflow[negative_inflow > 0] = 0  # 只保留负数
            
            # 绘制正数（绿色）
            add_plots.append(
                mpf.make_addplot(positive_inflow, panel=1, type='bar', color='green', 
                               alpha=0.6, width=0.8, ylabel='净流入', secondary_y=False)
            )
            
            # 绘制负数（红色）
            add_plots.append(
                mpf.make_addplot(negative_inflow, panel=1, type='bar', color='red', 
                               alpha=0.6, width=0.8, secondary_y=False)
            )
        
        # 添加基于net_inflow的OBV类似指标（只计算最近3天的数据）
        if 'net_inflow' in df.columns:
            # 只使用最近3天的数据来计算OBV指标
            df_with_timestamp = df.copy()
            df_with_timestamp['timestamp'] = pd.to_datetime(df_with_timestamp['timestamp'])
            latest_time = df_with_timestamp['timestamp'].max()
            three_days_ago = latest_time - pd.Timedelta(days=self.history_periods[interval])
            df_3days = df_with_timestamp[df_with_timestamp['timestamp'] >= three_days_ago].copy()
            
            if not df_3days.empty and 'net_inflow' in df_3days.columns:
                # 计算OBV指标（基于最近3天的数据，从3天前的第一个值开始累积）
                obv_series_3days = self.calculate_net_inflow_obv(df_3days)
                
                # 创建完整的OBV序列（3天前的数据设为NaN，3天内的数据使用计算值）
                df_indexed = df.set_index('timestamp')
                obv_full_series = pd.Series(index=df_indexed.index, dtype=float)
                df_3days_indexed = df_3days.set_index('timestamp')
                
                # 确保索引对齐
                for idx in df_3days_indexed.index:
                    if idx in obv_full_series.index:
                        obv_idx = df_3days_indexed.index.get_loc(idx)
                        obv_full_series.loc[idx] = obv_series_3days.iloc[obv_idx]
                
                # 添加到panel 1（使用折线图，左侧y轴，与net_inflow共享）
                add_plots.append(
                    mpf.make_addplot(obv_full_series, panel=1, color='darkorange', width=1.5, 
                                   label='Net Inflow OBV', secondary_y=False)
                )
        
        # 添加MFI指标到panel 1（使用折线图，右侧y轴）
        if 'mfi' in df.columns:
            mfi_series = df.set_index('timestamp')['mfi']
            add_plots.append(
                mpf.make_addplot(mfi_series, panel=1, color='royalblue', width=1.2, 
                               ylabel='MFI', secondary_y=True)
            )
            add_plots.append(
                mpf.make_addplot(pd.Series(80, index=mfi_series.index), panel=1,
                                 color='purple', linestyle='--', width=0.8, 
                                secondary_y=True)
            )
            add_plots.append(
                mpf.make_addplot(pd.Series(20, index=mfi_series.index), panel=1,
                                 color='orange', linestyle='--', width=0.8, 
                                 secondary_y=True)
            )
        
        market_colors = mpf.make_marketcolors(up='green', down='red', inherit=True)
        mpf_style = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=market_colors)

        save_path = f"{symbol}_{interval}_kline_mfi.png"
        
        # 关闭交互模式，防止显示图片，同时获取Figure和Axes进行自定义标注
        plt.ioff()
        
        fig, axes = mpf.plot(
            plot_df,
            type='candle',
            style=mpf_style,
            volume=False,  # 不在主面板显示volume
            addplot=add_plots if add_plots else None,
            title=f'{symbol} {interval} K线 & 净流入 & MFI',
            figsize=(16, 10),
            panel_ratios=(3, 1) if add_plots else None,
            xrotation=20,
            tight_layout=True,
            datetime_format='%Y-%m-%d %H:%M',
            ylabel='价格 (USDT)',
            returnfig=True
        )

        # 保存图片
        fig.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0.25)
        plt.close(fig)
        
        print(f"图表已保存到: {save_path}")
    
    def load_history_from_csv(self) -> pd.DataFrame:
        """
        从CSV文件加载历史K线和MFI数据
        
        Returns:
            pandas.DataFrame: 历史数据，如果文件不存在或为空则返回空DataFrame
        """
        if not os.path.exists(self.csv_file_path):
            print(f"CSV文件不存在: {self.csv_file_path}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(self.csv_file_path)
            if df.empty:
                print(f"CSV文件为空: {self.csv_file_path}")
                return pd.DataFrame()
            
            # 转换时间戳为datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 确保数值列为数值类型
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'net_inflow', 'mfi']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 按时间排序
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            print(f"从CSV文件加载了 {len(df)} 条历史记录")
            print(f"时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
            
            return df
        except Exception as e:
            print(f"读取CSV文件失败: {e}")
            return pd.DataFrame()
    
    def save_to_csv(self, df: pd.DataFrame):
        """
        保存K线和MFI数据到CSV文件
        
        Args:
            df: 包含K线和MFI数据的DataFrame
        """
        try:
            # 确保包含所有必要的列
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                print("数据缺少必要的列，无法保存")
                return
            
            # 选择要保存的列
            save_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if 'quote_volume' in df.columns:
                save_cols.append('quote_volume')
            if 'net_inflow' in df.columns:
                save_cols.append('net_inflow')
            if 'mfi' in df.columns:
                save_cols.append('mfi')
            
            # 按时间排序并去重（保留最新的记录）
            df_save = df[save_cols].copy()
            df_save = df_save.sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='last')
            
            # 保存到CSV
            df_save.to_csv(self.csv_file_path, index=False)
            print(f"数据已保存到 {self.csv_file_path}，共 {len(df_save)} 条记录")
        except Exception as e:
            print(f"保存CSV文件失败: {e}")



def send_dingtalk_message(webhook_url: str, message: str, title: str = "交易信号提醒"):
    """
    发送钉钉消息
    
    Args:
        webhook_url: 钉钉机器人webhook地址
        message: 消息内容
        title: 消息标题
    """
    if not webhook_url:
        print("未配置钉钉webhook地址，跳过消息发送")
        return False
    
    try:
        data = {
            "msgtype": "markdown",
            "markdown": {
                "title": title,
                "text": f"## {title}\n\n{message}"
            }
        }
        
        response = requests.post(webhook_url, json=data, timeout=10)
        response.raise_for_status()
        result = response.json()
        
        if result.get('errcode') == 0:
            print(f"钉钉消息发送成功: {title}")
            return True
        else:
            print(f"钉钉消息发送失败: {result.get('errmsg', '未知错误')}")
            return False
    except Exception as e:
        print(f"发送钉钉消息时出错: {e}")
        return False


def detect_trading_signals(df: pd.DataFrame, binance: BinanceKlineData, symbol: str, interval: str, 
                          dingtalk_webhook: Optional[str] = None) -> Optional[Dict]:
    """
    检测最新的交易信号（只检测最新的信号，避免重复检测历史信号）
    
    Args:
        df: 包含mfi和net_inflow的DataFrame
        binance: BinanceKlineData实例
        symbol: 交易对
        interval: 时间间隔
        dingtalk_webhook: 钉钉webhook地址（可选）
        
    Returns:
        Optional[Dict]: 检测到的最新交易信号，如果没有则返回None
    """
    if df.empty or 'mfi' not in df.columns or 'net_inflow' not in df.columns:
        print("数据不完整，无法检测交易信号")
        return None
    
    # 计算net inflow obv
    obv_series = binance.calculate_net_inflow_obv(df)
    if obv_series.empty:
        print("无法计算OBV指标，跳过信号检测")
        return None
    
    df_with_obv = df.copy()
    df_with_obv['net_inflow_obv'] = obv_series.values
    
    # 需要至少3条数据才能判断转折点（需要左右相邻的数据）
    if len(df_with_obv) < 3:
        print("数据不足，无法检测转折点")
        return None
    
    # 只检查最后几条K线（检查倒数第二条，因为最后一条可能没有下一个mfi值）
    # 检查范围：从倒数第3条到倒数第2条（确保有左右相邻数据）
    check_start = max(1, len(df_with_obv) - 3)
    check_end = len(df_with_obv) - 1
    
    # 先遍历历史数据，找到所有转折点，用于比较OBV值
    last_long_turn_point = None  # 上一个多头转折点
    last_short_turn_point = None  # 上一个空头转折点
    
    # 遍历历史数据（除了最后要检查的几条），记录转折点
    for i in range(1, check_start):
        if pd.isna(df_with_obv.iloc[i]['mfi']) or pd.isna(df_with_obv.iloc[i-1]['mfi']) or pd.isna(df_with_obv.iloc[i+1]['mfi']):
            continue
        
        current_mfi = df_with_obv.iloc[i]['mfi']
        prev_mfi = df_with_obv.iloc[i-1]['mfi']
        next_mfi = df_with_obv.iloc[i+1]['mfi']
        current_obv = df_with_obv.iloc[i]['net_inflow_obv']
        
        # 记录多头转折点
        if current_mfi < 30 and prev_mfi > current_mfi and next_mfi > current_mfi:
            last_long_turn_point = {
                'timestamp': df_with_obv.iloc[i]['timestamp'],
                'obv': current_obv
            }
        
        # 记录空头转折点
        elif current_mfi > 70 and prev_mfi < current_mfi and next_mfi < current_mfi:
            last_short_turn_point = {
                'timestamp': df_with_obv.iloc[i]['timestamp'],
                'obv': current_obv
            }
    
    # 只检查最新的几条K线，看是否有新的转折点
    for i in range(check_start, check_end):
        if pd.isna(df_with_obv.iloc[i]['mfi']) or pd.isna(df_with_obv.iloc[i-1]['mfi']) or pd.isna(df_with_obv.iloc[i+1]['mfi']):
            continue
        
        current_mfi = df_with_obv.iloc[i]['mfi']
        prev_mfi = df_with_obv.iloc[i-1]['mfi']
        next_mfi = df_with_obv.iloc[i+1]['mfi']
        current_obv = df_with_obv.iloc[i]['net_inflow_obv']
        
        # 检测多头转折点：mfi < 30，且左右相邻的mfi都大于当前mfi
        if current_mfi < 30 and prev_mfi > current_mfi and next_mfi > current_mfi:
            timestamp = df_with_obv.iloc[i]['timestamp']
            price = df_with_obv.iloc[i]['close']
            
            # 如果该转折点的net inflow obv值大于前一个转折点的net inflow obv值，则提示做多
            if last_long_turn_point is not None:
                last_obv = last_long_turn_point['obv']
                if current_obv > last_obv:
                    signal = {
                        'type': 'long',
                        'timestamp': timestamp,
                        'price': price,
                        'mfi': current_mfi,
                        'obv': current_obv,
                        'prev_obv': last_obv
                    }
                    
                    # 发送钉钉消息
                    message = (
                        f"**做多信号** 🟢\n\n"
                        f"交易对: {symbol}\n"
                        f"时间周期: {interval}\n"
                        f"时间: {timestamp}\n"
                        f"价格: {price:.4f} USDT\n"
                        f"MFI: {current_mfi:.2f}\n"
                        f"当前OBV: {current_obv:.2f}\n"
                        f"前一个转折点OBV: {last_obv:.2f}\n"
                        f"OBV增长: {current_obv - last_obv:.2f}"
                    )
                    
                    if dingtalk_webhook:
                        send_dingtalk_message(dingtalk_webhook, message, f"{symbol} 做多信号")
                    else:
                        print(f"\n{'='*50}")
                        print(f"做多信号: {symbol} {interval}")
                        print(message)
                        print(f"{'='*50}\n")
                    
                    return signal
        
        # 检测空头转折点：mfi > 70，且左右相邻的mfi都小于当前mfi
        elif current_mfi > 70 and prev_mfi < current_mfi and next_mfi < current_mfi:
            timestamp = df_with_obv.iloc[i]['timestamp']
            price = df_with_obv.iloc[i]['close']
            
            # 如果该转折点的net inflow obv值小于前一个转折点的net inflow obv值，则提示做空
            if last_short_turn_point is not None:
                last_obv = last_short_turn_point['obv']
                if current_obv < last_obv:
                    signal = {
                        'type': 'short',
                        'timestamp': timestamp,
                        'price': price,
                        'mfi': current_mfi,
                        'obv': current_obv,
                        'prev_obv': last_obv
                    }
                    
                    # 发送钉钉消息
                    message = (
                        f"**做空信号** 🔴\n\n"
                        f"交易对: {symbol}\n"
                        f"时间周期: {interval}\n"
                        f"时间: {timestamp}\n"
                        f"价格: {price:.4f} USDT\n"
                        f"MFI: {current_mfi:.2f}\n"
                        f"当前OBV: {current_obv:.2f}\n"
                        f"前一个转折点OBV: {last_obv:.2f}\n"
                        f"OBV下降: {last_obv - current_obv:.2f}"
                    )
                    
                    if dingtalk_webhook:
                        send_dingtalk_message(dingtalk_webhook, message, f"{symbol} 做空信号")
                    else:
                        print(f"\n{'='*50}")
                        print(f"做空信号: {symbol} {interval}")
                        print(message)
                        print(f"{'='*50}\n")
                    
                    return signal
    
    return None


def calculate_net_inflow_and_mfi(binance: BinanceKlineData, df: pd.DataFrame, symbol: str, interval: str):
    """
    计算K线数据的net_inflow和MFI值
    
    Args:
        binance: BinanceKlineData实例
        df: 包含K线数据的DataFrame
        symbol: 交易对
        interval: 时间间隔
        
    Returns:
        pd.DataFrame: 包含net_inflow和mfi的DataFrame
    """
    # 计算net_inflow
    if 'net_inflow' not in df.columns:
        df['net_inflow'] = np.nan
    
    # 根据interval确定每个K线的时间跨度（秒）
    interval_seconds_map = {
        '15m': 15 * 60,
        '1h': 3600,
        '4h': 4 * 3600,
        '1d': 86400
    }
    kline_seconds = interval_seconds_map.get(interval, 15 * 60)
    
    for i in range(df.index[0], df.index[-1] + 1):
        if pd.notna(df.loc[i, 'net_inflow']):
            # 如果已经有net_inflow值，跳过
            continue
            
        candlestick_df = df.iloc[i:i+1]
        start_dt = pd.to_datetime(candlestick_df['timestamp'].iloc[0]).to_pydatetime()
        end_dt = start_dt + timedelta(seconds=kline_seconds-1)
        
        one_second_df = binance.get_kline_data(symbol=symbol, interval="1s", start_time=start_dt, end_time=end_dt)
        if one_second_df.empty:
            print(f"未能获取 {symbol} 的 {start_dt} ~ {end_dt} 1秒K线数据")
            continue
        net_inflow = binance.calculate_typical_price_volume(one_second_df)
        print(f"timestamp: {start_dt}, net_inflow: {net_inflow}")
        df.loc[i, 'net_inflow'] = net_inflow
    
    # 计算MFI
    if 'mfi' not in df.columns:
        df['mfi'] = np.nan
    
    window = 14
    mfi_values = []
    for i in range(df.index[0], df.index[-1] + 1):
        if pd.notna(df.loc[i, 'mfi']):
            mfi_values.append(float(df.loc[i, 'mfi']))
            continue
        
        # 仅在有足够窗口时计算
        if i < window - 1 or 'net_inflow' not in df.columns or pd.isna(df.loc[i, 'net_inflow']):
            mfi_values.append(float('nan'))
            continue
        
        window_net_inflow = df['net_inflow'].iloc[i - window + 1:i + 1]

        # 正向为大于零、负向为小于零
        positive = window_net_inflow[window_net_inflow > 0].sum()
        negative = -window_net_inflow[window_net_inflow < 0].sum()  # 负数转正

        if negative == 0:
            mfi = 100.0  # 极端情况处理
        elif positive == 0:
            mfi = 0.0
        else:
            mfr = positive / negative
            mfi = 100 - (100 / (1 + mfr))
        mfi_values.append(mfi)
        print(f"timestamp: {df.loc[i, 'timestamp']}, positive: {positive}, negative: {negative}, mfi: {mfi}")
    
    df['mfi'] = mfi_values
    return df

def main():
    """主函数"""
    
    # 配置参数
    import argparse
    parser = argparse.ArgumentParser(description='币安K线净流入和MFI计算')
    parser.add_argument('--symbol', type=str, default="ETHUSDT", help='交易对，比如BTCUSDT, ETHUSDT')
    parser.add_argument('--interval', type=str, default="15m", help='时间间隔（如1m, 5m, 15m, 1h, 4h, 1d等）')
    parser.add_argument('--dingtalk-webhook', type=str, default=None, 
                       help='钉钉机器人webhook地址（可选），用于发送交易信号通知')
    args = parser.parse_args()
    symbol = args.symbol
    interval = args.interval
    dingtalk_webhook = args.dingtalk_webhook or os.getenv('DINGTALK_WEBHOOK')

        # 创建币安数据获取实例
    csv_file_path = f"kline_mfi_history_{symbol}_{interval}.csv"
    binance = BinanceKlineData(csv_file_path=csv_file_path)
    
    # 读取历史数据
    print("正在读取历史数据...")
    history_df = binance.load_history_from_csv()
    
    # 计算最近的整interval时间
    now = datetime.now()
    interval_seconds = binance.interval_seconds[interval]
    end_time = datetime.fromtimestamp((int(now.timestamp()) // interval_seconds) * interval_seconds)

    history_period = binance.history_periods[interval]
    
    if history_df.empty:
        print(f"\nCSV文件为空，初始化过去{history_period}天的数据...")
        start_time = end_time - timedelta(days=history_period)
        end_time = end_time - timedelta(seconds=1)
        
        print(f"正在获取 {symbol} 的 {interval} K线数据...")
        print(f"时间范围: {start_time} 到 {end_time}\n")
        
        df = binance.get_kline_data(
            symbol=symbol,
            interval=interval,
            start_time=start_time,
            end_time=end_time
        )
        
        if df.empty:
            print("未能获取数据，程序退出")
            return
        
        # 计算net_inflow和MFI
        df = calculate_net_inflow_and_mfi(binance, df, symbol, interval)
        
    else:
        # 如果不为空，读取文件中过去3天的历史值
        print(f"\nCSV文件不为空，读取过去{history_period}天的历史数据...")
        history_start_time = end_time - timedelta(days=3)

        # 确保timestamp是datetime类型
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
        
        # 统一时区为历史数据的timestamp的时区
        ts_tz = history_df['timestamp'].dt.tz
        recent_history = history_df[history_df['timestamp'] >= pd.Timestamp(history_start_time, tz=ts_tz)].copy()
        
        if recent_history.empty:
            print(f"历史数据中没有过去{history_period}天的记录，使用全部历史数据")
            recent_history = history_df.copy()
        else:
            print(f"从历史数据中读取了 {len(recent_history)} 条过去{history_period}天的记录")
        print(f"历史数据时间范围: {recent_history['timestamp'].min()} 到 {recent_history['timestamp'].max()}")
        
        # 确定需要补全的时间范围（基于全部历史数据的最新时间）
        latest_timestamp = history_df['timestamp'].max()
        new_df = pd.DataFrame()  # 初始化new_df
        fill_start_time = latest_timestamp + timedelta(seconds=interval_seconds)  # 从下一条开始
        fill_start_time = pd.to_datetime(fill_start_time).to_pydatetime().replace(tzinfo=None)
        
        if fill_start_time < end_time:
            # 需要补全从latest_timestamp到end_time的数据
            print(f"\n需要补全从 {fill_start_time} 到 {end_time} 的数据...")
            
            # 获取缺失时间段的数据
            
            end_time = end_time - timedelta(seconds=1)
            
            new_df = binance.get_kline_data(
                symbol=symbol,
                interval=interval,
                start_time=fill_start_time,
                end_time=end_time
            )
            
            if not new_df.empty:
                print(f"获取了 {len(new_df)} 条新数据")
                df_for_calc = pd.concat([recent_history, new_df], ignore_index=True)
                df_for_calc = df_for_calc.sort_values('timestamp').reset_index(drop=True)
                # 去重，保留最新的
                df_for_calc = df_for_calc.drop_duplicates(subset=['timestamp'], keep='last')
            else:
                print("未能获取新数据，使用历史数据")
                df_for_calc = recent_history.copy()
        else:
            # 历史数据已经是最新的，直接使用
            print("历史数据已是最新，无需补全")
            df_for_calc = recent_history.copy()
        
        # 计算缺失的net_inflow和MFI（基于过去7天+新数据）
        df_for_calc = calculate_net_inflow_and_mfi(binance, df_for_calc, symbol, interval)
        
        # 将计算结果合并回完整的历史数据
        # 使用merge更新历史数据中已有的记录，并添加新记录
        if 'net_inflow' in df_for_calc.columns or 'mfi' in df_for_calc.columns:
            # 使用merge更新net_inflow和mfi
            update_cols = ['timestamp']
            if 'net_inflow' in df_for_calc.columns:
                update_cols.append('net_inflow')
            if 'mfi' in df_for_calc.columns:
                update_cols.append('mfi')
            
            # 更新历史数据中已有的记录
            history_df = history_df.merge(
                df_for_calc[update_cols], 
                on='timestamp', 
                how='left', 
                suffixes=('', '_new')
            )
            
            # 用新值替换旧值
            if 'net_inflow_new' in history_df.columns:
                history_df['net_inflow'] = history_df['net_inflow_new'].fillna(history_df['net_inflow'])
                history_df = history_df.drop(columns=['net_inflow_new'])
            if 'mfi_new' in history_df.columns:
                history_df['mfi'] = history_df['mfi_new'].fillna(history_df['mfi'])
                history_df = history_df.drop(columns=['mfi_new'])
        
        # 添加新记录（在df_for_calc中但不在history_df中）
        existing_timestamps = set(history_df['timestamp'])
        new_records = df_for_calc[~df_for_calc['timestamp'].isin(existing_timestamps)]
        
        if not new_records.empty:
            # 确保新记录包含所有必要的列
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in new_records.columns and col in df_for_calc.columns:
                    pass  # 列已存在
                elif col not in new_records.columns:
                    # 如果新记录缺少必要的列，尝试从df_for_calc获取
                    if col in df_for_calc.columns:
                        new_records[col] = df_for_calc.loc[new_records.index, col]
            
            # 合并新记录
            df = pd.concat([history_df, new_records], ignore_index=True)
        else:
            df = history_df.copy()
        
        # 按时间排序并去重
        df = df.sort_values('timestamp').reset_index(drop=True)
        df = df.drop_duplicates(subset=['timestamp'], keep='last')
    
    print(f"\n最终数据预览:")
    print(df.head())
    print(f"\n数据统计:")
    print(df.describe())
    
    # 保存到CSV
    print("\n正在保存数据到CSV...")
    binance.save_to_csv(df)
    
    # 使用mplfinance绘制更美观的图表
    print("\n正在生成图表...")
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        latest_time = df['timestamp'].max()
        history_period_time = latest_time - pd.Timedelta(days=history_period)
        df_last = df[df['timestamp'] >= history_period_time].copy()
    else:
        df_last = df.copy()
    binance.plot_with_mplfinance(df_last, symbol, interval)
    
    # 检测最新的交易信号（只检测最新信号，避免重复检测历史信号）
    print("\n正在检测最新交易信号...")
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # 使用足够的历史数据来找到前一个转折点进行比较
        # 但只检测最新的几条K线
        latest_time = df['timestamp'].max()
        signal_check_time = latest_time - pd.Timedelta(days=7)
        df_for_signal = df[df['timestamp'] >= signal_check_time].copy()
    else:
        df_for_signal = df.copy()
    
    if not df_for_signal.empty and 'mfi' in df_for_signal.columns and 'net_inflow' in df_for_signal.columns:
        signal = detect_trading_signals(df_for_signal, binance, symbol, interval, dingtalk_webhook)
        if signal:
            print(f"\n检测到最新交易信号: {signal['type']} 信号")
        else:
            print("未检测到最新交易信号")
    else:
        print("数据不完整，跳过交易信号检测")


if __name__ == "__main__":
    main()

