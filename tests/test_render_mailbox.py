import numpy as np

from scripts.render_mailbox import LatestState


def test_latest_state_keeps_only_the_newest_snapshot():
    mailbox = LatestState()
    mailbox.publish(np.array([1.0, 2.0]))
    mailbox.publish(np.array([3.0, 4.0]))

    np.testing.assert_array_equal(mailbox.latest(), np.array([3.0, 4.0]))


def test_latest_state_returns_a_copy_that_cannot_mutate_mailbox():
    mailbox = LatestState()
    mailbox.publish(np.array([1.0, 2.0]))
    snapshot = mailbox.latest()
    snapshot[0] = 99.0

    np.testing.assert_array_equal(mailbox.latest(), np.array([1.0, 2.0]))
