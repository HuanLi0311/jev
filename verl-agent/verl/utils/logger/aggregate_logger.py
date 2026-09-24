# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A Ray logger will receive logging info from different processes.
"""

import json
import logging
import numbers
import os
import time
from pathlib import Path
from typing import Dict


def concat_dict_to_str(dict: Dict, step):
    output = [f"step:{step}"]
    for k, v in dict.items():
        if isinstance(v, numbers.Number):
            output.append(f"{k}:{v:.3f}")
    output_str = " - ".join(output)
    return output_str


class LocalLogger:
    def __init__(self, remote_logger=None, enable_wandb=False, print_to_console=False):
        self.print_to_console = print_to_console
        metrics_file = os.environ.get("VERL_METRICS_FILE")
        self.metrics_file = Path(metrics_file) if metrics_file else None
        self.started = time.monotonic()
        if self.metrics_file:
            self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        if print_to_console:
            print("Using LocalLogger is deprecated. The constructor API will change ")

    def flush(self):
        pass

    def log(self, data, step):
        if self.metrics_file:
            record = {
                "step": int(step),
                "time_unix": time.time(),
                "process_elapsed_seconds": time.monotonic() - self.started,
            }
            record.update({
                key: value.item() if hasattr(value, "item") else value
                for key, value in data.items()
                if isinstance(value, numbers.Number)
            })
            with self.metrics_file.open("a", encoding="utf-8") as sink:
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
        if self.print_to_console:
            print(concat_dict_to_str(data, step=step), flush=True)


class DecoratorLoggerBase:
    def __init__(self, role: str, logger: logging.Logger = None, level=logging.DEBUG, rank: int = 0, log_only_rank_0: bool = True):
        self.role = role
        self.logger = logger
        self.level = level
        self.rank = rank
        self.log_only_rank_0 = log_only_rank_0
        self.logging_function = self.log_by_logging
        if logger is None:
            self.logging_function = self.log_by_print

    def log_by_print(self, log_str):
        if not self.log_only_rank_0 or self.rank == 0:
            print(f"{self.role} {log_str}", flush=True)

    def log_by_logging(self, log_str):
        if self.logger is None:
            raise ValueError("Logger is not initialized")
        if not self.log_only_rank_0 or self.rank == 0:
            self.logger.log(self.level, f"{self.role} {log_str}")
