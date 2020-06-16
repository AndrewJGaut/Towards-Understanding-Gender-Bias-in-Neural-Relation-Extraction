'''
Wikigender format:
{
'train':
 [ {
    entity1: NAME,
    relations: [
        {
            relation_name: spouse,
            entity2: NAME,
            sentences: [
            ]
        }
    ]
} ],
'dev' : [ {
    entity1: NAME,
    relations: [
        {
            relation_name: spouse,
            entity2: NAME,
            sentences: [
            ]
        }
    ]
}, ... ], ...
}
'''
from __future__ import absolute_import
import unittest

from DebiasJsonDataset import *


class UnitTests(unittest.TestCase):
    def test_json_genderswap(self):
        self.maxDiff = None

        # set up test data
        input_data = {
            'train': [
                {
                    'gender_of_entity1': 'Male',
                    'entity1': 'test1',
                    'relations': [
                        {
                            'relation_name': 'spouse',
                            'entity2': 'Johnny',
                            'sentences': [
                                'Johnny and his mother were talking to her father on the sister.',
                                'Marge was with the actress, the boy, the girl, and the businessman Mr. Johnson, she told my aunt.',

                            ],
                            'positions':
                            [
                                {'entity1': [0,0], 'entity2': [0,7]},
                                {'entity1': [0,0], 'entity2': [0,0]}
                            ]
                        },

                    ]
                }
            ],
            'dev': [

            ],
            'male_test': [

            ],
            'female_test': [

            ]
        }

        expected_output = {
            'train': [
                {
                    'gender_of_entity1': 'Male',
                    'entity1': 'test1',
                    'relations': [
                        {
                            'relation_name': 'spouse',
                            'entity2': 'Johnny',
                            'sentences': [
                                'Johnny and his mother were talking to her father on the sister.',
                                'Marge was with the actress, the boy, the girl, and the businessman Mr. Johnson, she told my aunt.',

                            ],
                            'positions':
                                [
                                    {'entity1': [0, 0], 'entity2': [0, 7]},
                                    {'entity1': [0, 0], 'entity2': [0, 0]}
                                ]
                        },

                    ]
                },
                {
                    'gender_of_entity1': 'Female',
                    'entity1': 'test1',
                    'relations': [
                        {
                            'relation_name': 'spouse',
                            'entity2': 'Johnny',
                            'sentences': [
                                'Johnny and her father were talking to his mother on the brother .',
                                'Marge was with the actor , the girl , the guy , and the businesswoman ms Johnson , he told my uncle .'
                            ],
                            'positions':
                            [
                                {'entity1': [0,0], 'entity2': [0,7]},
                                {'entity1': [80,83], 'entity2': [0,0]} # note entity 1 has a position due to soft matching
                            ]
                        }
                    ]
                }
            ],
            'dev': [

            ],
            'male_test': [

            ],
            'female_test': [

            ]
        }

        x = createGenderSwappedDatasetEntries(input_data, swap_names=False, neutralized=False)
        self.assertEqual(x, expected_output)


    @unittest.skip("We need to add position data and gender data to make this a valid test")
    def testJsonNameAnonynize(self):
        self.maxDiff = None

        # set up test data
        input_data = {
            'train': [
                {
                    'entity1': 'testone',
                    'relations': [
                        {
                            'relation_name': 'spouse',
                            'entity2': 'Johnny',
                            'sentences': [
                                'Johnny and his mother were talking to her father on the sister.',
                                'Marge was with the actress, the boy, the girl, and the businessman Mr. Johnson, she told my aunt.'
                            ]
                        }

                    ]
                }
            ],
            'dev': [
                {
                    'entity1': 'testtwo',
                    'relations': [
                        {
                            'relation_name': 'birthdate',
                            'entity2': '09/18/19',
                            'sentences': [
                                'Jeff saw testtwo with his mom Marge on 09/18/19; it was cool.',
                            ]
                        }

                    ]
                }

            ],
            'male_test': [

            ],
            'female_test': [

            ]
        }

        expected_output = {
            'train': [
                {
                    'entity1': 'E0',
                    'relations': [
                        {
                            'relation_name': 'spouse',
                            'entity2': 'E1',
                            'sentences': [
                                'E1 and his mother were talking to her father on the sister .',
                                'E2 was with the actress , the boy , the girl , and the businessman Mr . Johnson, she told my aunt .'
                            ]
                        }
                    ]
                }
            ],
            'dev': [
                {
                    'entity1': 'E3',
                    'relations': [
                        {
                            'relation_name': 'birthdate',
                            'entity2': '09/18/19',
                            'sentences': [
                                'E4 saw E3 with his mom E2 on 09 / 18 / 19 ; it was cool .',
                            ]
                        }

                    ]
                }
            ],
            'male_test': [

            ],
            'female_test': [

            ]
        }

        x = createNameAnonymizedJsonDatasetEntries(input_data)
        self.assertEqual(x, expected_output)


    @unittest.skip("We need to add position data and gender data to make this a valid test")
    def testJsonNameAnonynizeAndGenderSwap(self):
        self.maxDiff = None

        # set up test data
        input_data = {
            'train': [
                {
                    'entity1': 'test1',
                    'relations': [
                        {
                            'relation_name': 'spouse',
                            'entity2': 'Johnny',
                            'sentences': [
                                'Johnny and his mother were talking to her father on the sister.',
                                'Marge was with the actress, the boy, the girl, and the businessman Mr. Johnson, she told my aunt.'
                            ]
                        }

                    ]
                }
            ],
            'dev': [
                {
                    'entity1': 'testtwo',
                    'relations': [
                        {
                            'relation_name': 'birthdate',
                            'entity2': '09/18/19',
                            'sentences': [
                                'Jeff saw testtwo with his mom Marge on 09/18/19; it was cool.',
                            ]
                        }

                    ]
                },
                {
                    'entity1': 'testthree',
                    'relations': [
                        {
                            'relation_name': 'spouse',
                            'entity2': 'Johnny',
                            'sentences': [
                                'testthree and Johnny were with the businessmen',
                            ]
                        }

                    ]
                }

            ],
            'male_test': [

            ],
            'female_test': [

            ]
        }

        expected_output = {
            'train': [
                {
                    'entity1': 'E0',
                    'relations': [
                        {
                            'relation_name': 'spouse',
                            'entity2': 'E1',
                            'sentences': [
                                'E1 and her father were talking to his mother on the brother.',
                                'E2 was with the actor, the girl, the guy, and the businesswoman ms Johnson, he told my uncle.'
                            ]
                        }
                    ]
                }
            ],
            'dev': [
                {
                    'entity1': 'E3',
                    'relations': [
                        {
                            'relation_name': 'birthdate',
                            'entity2': '09/18/19',
                            'sentences': [
                                'E4 saw E3 with her dad E2 on 09/18/19; it was cool.',
                            ]
                        }

                    ]
                },
                {
                    'entity1': 'E5',
                    'relations': [
                        {
                            'relation_name': 'spouse',
                            'entity2': 'E1',
                            'sentences': [
                                'E5 and E1 were with the businesswomen',
                            ]
                        }

                    ]
                }
            ],
            'male_test': [

            ],
            'female_test': [

            ]
        }

        x = createNameAnonymizedJsonDatasetEntries(input_data)
        x = createGenderSwappedDatasetEntries(x)
        self.assertEqual(x, expected_output)
if __name__ == '__main__':
        unittest.main()
