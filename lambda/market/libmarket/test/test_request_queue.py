import libmarket.queue as queue
import unittest


class TestRequestQueue(unittest.TestCase):
    def test_add(self):
        q = queue.RequestQueue()
        request = queue.Request.from_string("voo", "2023-06")
        q.add(request)

    def test_pop(self):
        q = queue.RequestQueue()

        request = queue.Request.from_string("voo", "2023-06")
        q.add(request)

        output = q.pop()
        expectedOutput = queue.Request.from_string("voo", "2023-06")
        self.assertEqual(output, expectedOutput)

    def test_contains(self):
        q = queue.RequestQueue()
        request = queue.Request.from_string("voo", "2023-06")
        q.add(request)

        expectedOutput = queue.Request.from_string("voo", "2023-06")
        self.assertIn(expectedOutput, q)

    def test_multiple_requests(self):
        requests = [
             queue.Request.from_string("voo", "2023-06"),
             queue.Request.from_string("voo", "2023-07"),
             queue.Request.from_string("spy", "2012-04")
        ]

        q = queue.RequestQueue()
    
        for request in requests:
            q.add(request)

        for request in requests:
            self.assertIn(request, q)

        for original, stored in zip(requests, q):
            self.assertEqual(original, stored)

        output = q.pop()
        expectedOutput = queue.Request.from_string("voo", "2023-06")
        self.assertEqual(output, expectedOutput)

    def test_serialize(self):
        requests = [
             queue.Request.from_string("voo", "2023-06"),
             queue.Request.from_string("voo", "2023-07"),
             queue.Request.from_string("spy", "2012-04")
        ]

        q = queue.RequestQueue()
    
        for request in requests:
            q.add(request)

        serialized_queue = q.serialize()
        reconstructed_queue = queue.RequestQueue.deserialize(serialized_queue)

        self.assertEqual(q.requests, reconstructed_queue.requests)
        for key in q.index:
            self.assertIn(key, reconstructed_queue.index)
            self.assertSetEqual(q.index[key], reconstructed_queue.index[key])


    def test_bad_deserialize(self):
        serializations = [
            "",
            "[]",
            "asdlkfjasdfasd",
            '["test", 1, 2.4]',
            '[{"symbol": "VOO"}, {"price": 2.45}]',
            '[{"symbol": "VOO", "month": "2023-07"}, {"symbol": "VOO"}]'
            '[{"symbol": "VOO", "month": "2023-07"}, {"symbol": "VOO", "month": "2023-06"}]'
        ] 

        is_correct_serialization = [False, True, False, False, False, False, True]

        for serialization, correct in zip(serializations, is_correct_serialization):
            if correct:
                queue.RequestQueue.deserialize(serialization)
            else:
                with self.assertRaises(ValueError):
                    queue.RequestQueue.deserialize(serialization)
