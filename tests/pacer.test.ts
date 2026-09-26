import { describe, it, expect } from 'vitest';
import { IndexingPace } from '../src/embedding/pacer.js';

/** Records rests instead of taking them, so the test costs no time. */
function recordingPace(dutyPercent: number, workSliceMs: number) {
  const rests: number[] = [];
  const pace = new IndexingPace({
    dutyPercent,
    workSliceMs,
    sleep: async (ms) => {
      rests.push(ms);
    },
  });
  return { pace, rests };
}

describe('Indexing pace', () => {
  it('Background_work_rests_after_each_slice_so_the_machine_stays_usable', async () => {
    const { pace, rests } = recordingPace(20, 100);

    await pace.afterWorking(100);

    // A fifth of the time working means four times the work spent resting.
    expect(rests).toEqual([400]);
  });

  it('Work_shorter_than_a_slice_does_not_rest_until_it_adds_up', async () => {
    const { pace, rests } = recordingPace(20, 100);

    await pace.afterWorking(30);
    await pace.afterWorking(30);
    expect(rests).toEqual([]);

    await pace.afterWorking(40);
    expect(rests).toEqual([400]);
  });

  it('A_rest_starts_a_new_slice_rather_than_resting_on_every_call_after_the_first', async () => {
    const { pace, rests } = recordingPace(50, 100);

    await pace.afterWorking(100);
    await pace.afterWorking(50);
    expect(rests).toEqual([100]);

    await pace.afterWorking(50);
    expect(rests).toEqual([100, 100]);
  });

  it('Work_done_while_a_question_is_being_answered_never_rests', async () => {
    const { pace, rests } = recordingPace(20, 100);

    await pace.whileAnswering(async () => {
      await pace.afterWorking(1000);
      await pace.afterWorking(1000);
    });

    expect(rests).toEqual([]);
  });

  it('The_background_pace_returns_once_the_last_answer_is_given', async () => {
    const { pace, rests } = recordingPace(20, 100);

    await pace.whileAnswering(async () => {
      await pace.whileAnswering(async () => {
        await pace.afterWorking(1000);
      });
      await pace.afterWorking(1000);
    });
    expect(rests).toEqual([]);

    await pace.afterWorking(100);
    expect(rests).toEqual([400]);
  });

  it('Work_done_for_a_question_is_not_charged_to_the_next_background_slice', async () => {
    const { pace, rests } = recordingPace(20, 100);

    await pace.afterWorking(90);
    await pace.whileAnswering(async () => {
      await pace.afterWorking(500);
    });

    // The 90ms before the question is spent, so the next slice starts from nothing.
    await pace.afterWorking(90);
    expect(rests).toEqual([]);
  });

  it('A_failed_request_still_hands_the_pace_back_to_the_background', async () => {
    const { pace, rests } = recordingPace(20, 100);

    await expect(
      pace.whileAnswering(async () => {
        throw new Error('tool failed');
      })
    ).rejects.toThrow('tool failed');

    await pace.afterWorking(100);
    expect(rests).toEqual([400]);
  });
});
