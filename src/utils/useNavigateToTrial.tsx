import { useCallback } from 'react';
import { encryptIndex } from './encryptDecryptIndex';
import { PREFIX, sameOriginHref } from './Prefix';

export function useNavigateToTrial() {
  return useCallback((trialOrder: string, participantId: string, studyId: string) => {
    const rejoined = trialOrder.split('_').map((index) => encryptIndex(+index)).join('/');
    const url = `${studyId}/${rejoined}?participantId=${participantId}`;
    window.open(sameOriginHref(`${PREFIX}${url}`), '_blank');
  }, []);
}
