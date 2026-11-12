import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { summarize } from '../src/domain.js';

const seed = JSON.parse(await readFile('data/records.json', 'utf8')).records[0];
const probe = patches => summarize(patches.map((patch, index) => ({...seed, id: `p${index}`, ...patch}))).groups[0];

test('同企业多条商机不扩大企业清点数', () => {

  assert.equal(probe([{company: 'a'}, {company: 'a'}, {company: 'b'}]).companies, 2);
  assert.equal(probe([{company: 'a'}]).companies, 1);
});
