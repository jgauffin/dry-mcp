/**
 * How much of the time background indexing is allowed to be working.
 *
 * A developer keeps one of these servers per open project, so several index at
 * once on the machine they are also compiling and editing on. At a fifth of the
 * time, all of them together cost less than a core while nobody is waiting, and
 * a project still finishes indexing within an editor session.
 */
const BACKGROUND_DUTY_PERCENT = 20;

/**
 * How long background work may run before it rests.
 *
 * Short enough that the machine never feels held, long enough that the rests do
 * not cost more in scheduling than the work between them.
 */
const WORK_SLICE_MS = 250;

export interface PaceSettings {
  /** Share of the time background work may run, from 1 to 100. */
  dutyPercent?: number;
  /** How much work is done between rests. */
  workSliceMs?: number;
  /** How resting is done. Replaced in tests so they do not sleep. */
  sleep?: (ms: number) => Promise<void>;
}

/**
 * Paces indexing so the developer's machine stays theirs.
 *
 * Embedding a project is minutes of solid CPU, and nobody asked for it: it
 * happens because the code changed. Left to run flat out it takes every core the
 * model can use, which on a machine with several projects open is the whole
 * machine. So background work runs in slices with a rest between them.
 *
 * A question is different — someone is waiting for the answer — so work done
 * while one is being answered is never paced.
 */
export class IndexingPace {
  private dutyPercent: number;
  private workSliceMs: number;
  private sleep: (ms: number) => Promise<void>;
  /** Work done since the last rest, which decides how long the next one is. */
  private workedMs = 0;
  /** How many questions are being answered right now. */
  private answering = 0;

  constructor(settings: PaceSettings = {}) {
    this.dutyPercent = Math.min(100, Math.max(1, settings.dutyPercent ?? BACKGROUND_DUTY_PERCENT));
    this.workSliceMs = settings.workSliceMs ?? WORK_SLICE_MS;
    this.sleep = settings.sleep ?? realSleep;
  }

  /**
   * Runs work at full speed, because a caller is waiting for it.
   *
   * Nested calls count, so indexing only goes back to its background pace once
   * the last question has been answered.
   */
  async whileAnswering<T>(work: () => Promise<T>): Promise<T> {
    this.answering++;
    try {
      return await work();
    } finally {
      this.answering--;
      // Work done for the question is not charged to the background budget, so
      // indexing does not begin its next slice already owing a rest.
      this.workedMs = 0;
    }
  }

  /**
   * Rests, if the work just reported has used up a slice.
   *
   * Called with the time a piece of indexing took, rather than measuring it
   * here, so that the pause lands between pieces of work and never inside one.
   */
  async afterWorking(elapsedMs: number): Promise<void> {
    if (this.answering > 0) {
      this.workedMs = 0;
      return;
    }

    this.workedMs += Math.max(0, elapsedMs);
    if (this.workedMs < this.workSliceMs) return;

    const worked = this.workedMs;
    this.workedMs = 0;
    await this.sleep(restFor(worked, this.dutyPercent));
  }
}

/** How long to rest so that `worked` milliseconds amount to the allowed share. */
function restFor(workedMs: number, dutyPercent: number): number {
  return Math.round((workedMs * (100 - dutyPercent)) / dutyPercent);
}

function realSleep(ms: number): Promise<void> {
  if (ms <= 0) return Promise.resolve();
  return new Promise((resolve) => setTimeout(resolve, ms));
}
