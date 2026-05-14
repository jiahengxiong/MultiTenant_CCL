#!/usr/bin/env python3
import argparse
import csv
from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass
class Event:
    row_index: int
    comm_type: str
    stage: str
    elapsed_ms: Optional[float]

    @property
    def is_collective(self) -> bool:
        return self.comm_type.startswith("CommType.") and self.comm_type not in {
            "CommType.computation",
            "CommType.epoch_end",
        }


def _parse_elapsed_ms(value: str) -> Optional[float]:
    if value is None:
        return None
    text = value.strip()
    if not text or text == "None":
        return None
    return float(text)


def load_events(csv_path: str) -> List[Event]:
    events: List[Event] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row_index, row in enumerate(reader, start=1):
            events.append(
                Event(
                    row_index=row_index,
                    comm_type=(row.get("comm_type") or "").strip(),
                    stage=(row.get("stage") or "").strip(),
                    elapsed_ms=_parse_elapsed_ms(row.get("_elapsed_time", "")),
                )
            )
    return events


def select_collective(events: Iterable[Event], occurrence: int, comm_type: Optional[str]) -> Event:
    matches = [
        event
        for event in events
        if event.is_collective and (comm_type is None or event.comm_type == comm_type)
    ]
    if occurrence < 1 or occurrence > len(matches):
        qualifier = comm_type or "any collective"
        raise ValueError(
            f"Requested occurrence {occurrence} for {qualifier}, "
            f"but only found {len(matches)} matching collective events."
        )
    return matches[occurrence - 1]


def sum_gap_ms(events: List[Event], start_row: int, end_row: int) -> Optional[float]:
    gap_slice = events[start_row:end_row - 1]
    elapsed_values = [event.elapsed_ms for event in gap_slice]
    if any(value is None for value in elapsed_values):
        return None
    return sum(elapsed_values) if elapsed_values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the elapsed-time gap between two collective events in an AICB CSV. "
            "For runtime comm logs, the gap is the sum of _elapsed_time of all events "
            "strictly between the two selected collectives."
        )
    )
    parser.add_argument("--csv", required=True, help="Path to an AICB CSV file.")
    parser.add_argument(
        "--start-row",
        type=int,
        help="1-based data row index of the starting collective (excluding header).",
    )
    parser.add_argument(
        "--end-row",
        type=int,
        help="1-based data row index of the ending collective (excluding header).",
    )
    parser.add_argument(
        "--start-occurrence",
        type=int,
        help="Select the N-th collective occurrence as the start event.",
    )
    parser.add_argument(
        "--end-occurrence",
        type=int,
        help="Select the N-th collective occurrence as the end event.",
    )
    parser.add_argument(
        "--comm-type",
        help=(
            "Optional collective type filter when using occurrence selection, "
            'for example "CommType.all_reduce".'
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List collective rows and exit.",
    )
    parser.add_argument(
        "--all-adjacent",
        action="store_true",
        help=(
            "Print the gap between every adjacent pair of collective events in the CSV. "
            "This is intended for runtime comm logs."
        ),
    )
    args = parser.parse_args()

    events = load_events(args.csv)
    collectives = [event for event in events if event.is_collective]

    if args.list:
        for idx, event in enumerate(collectives, start=1):
            elapsed_str = "None" if event.elapsed_ms is None else f"{event.elapsed_ms:.6f}"
            print(
                f"occurrence={idx:4d} row={event.row_index:4d} "
                f"type={event.comm_type:24s} stage={event.stage:40s} elapsed_ms={elapsed_str}"
            )
        return

    if args.all_adjacent:
        for idx in range(len(collectives) - 1):
            start_event = collectives[idx]
            end_event = collectives[idx + 1]
            gap_ms = sum_gap_ms(events, start_event.row_index, end_event.row_index)
            gap_str = "UNKNOWN" if gap_ms is None else f"{gap_ms:.6f}"
            print(
                f"pair={idx + 1:4d} "
                f"start_row={start_event.row_index:4d} start_type={start_event.comm_type:24s} start_stage={start_event.stage:40s} "
                f"end_row={end_event.row_index:4d} end_type={end_event.comm_type:24s} end_stage={end_event.stage:40s} "
                f"events_between={end_event.row_index - start_event.row_index - 1:4d} gap_ms={gap_str}"
            )
        if len(collectives) < 2:
            print("note=Fewer than two collective events were found.")
        return

    if args.start_row is not None or args.end_row is not None:
        if args.start_row is None or args.end_row is None:
            raise ValueError("`--start-row` and `--end-row` must be provided together.")
        start_event = events[args.start_row - 1]
        end_event = events[args.end_row - 1]
    else:
        if args.start_occurrence is None or args.end_occurrence is None:
            raise ValueError(
                "Provide either (`--start-row`, `--end-row`) or "
                "(`--start-occurrence`, `--end-occurrence`)."
            )
        start_event = select_collective(events, args.start_occurrence, args.comm_type)
        end_event = select_collective(events, args.end_occurrence, args.comm_type)

    if start_event.row_index >= end_event.row_index:
        raise ValueError("Start event must appear before end event.")

    gap_ms = sum_gap_ms(events, start_event.row_index, end_event.row_index)

    print(f"start_row={start_event.row_index}")
    print(f"start_type={start_event.comm_type}")
    print(f"start_stage={start_event.stage}")
    print(f"end_row={end_event.row_index}")
    print(f"end_type={end_event.comm_type}")
    print(f"end_stage={end_event.stage}")
    print(f"events_between={end_event.row_index - start_event.row_index - 1}")

    if gap_ms is None:
        print("gap_ms=UNKNOWN")
        print(
            "note=At least one intermediate event has _elapsed_time=None. "
            "This usually means you passed a mocked workload CSV instead of a runtime comm log."
        )
    else:
        print(f"gap_ms={gap_ms:.6f}")


if __name__ == "__main__":
    main()
