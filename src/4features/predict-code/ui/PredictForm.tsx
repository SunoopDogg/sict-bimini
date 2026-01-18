'use client';

import { useActionState } from 'react';
import { useFormStatus } from 'react-dom';

import { type PredictState, predictAction } from '@/app/predict/actions';
import { Button } from '@/src/6shared/ui/primitive/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/src/6shared/ui/primitive/card';
import { Input } from '@/src/6shared/ui/primitive/input';
import { Label } from '@/src/6shared/ui/primitive/label';

interface PredictFormProps {
  onResult?: (state: PredictState) => void;
}

function SubmitButton() {
  const { pending } = useFormStatus();

  return (
    <Button type="submit" disabled={pending} className="w-full">
      {pending ? '예측 중...' : '부위코드 예측'}
    </Button>
  );
}

const formFields = [
  { name: 'ifc_type', label: 'IFC Type', placeholder: 'IfcBeam' },
  { name: 'category', label: '카테고리', placeholder: '구조 프레임' },
  {
    name: 'family_name',
    label: '패밀리 이름',
    placeholder: 'BM111-철근콘크리트구조 보',
  },
  { name: 'family', label: '패밀리', placeholder: 'BM111-철근콘크리트구조 보' },
  {
    name: 'type',
    label: '타입',
    placeholder: 'BM110-철근 콘크리트구조 보_400 x 850_B1-1G3',
  },
  { name: 'type_id', label: '타입 ID', placeholder: '7667203' },
];

export function PredictForm({ onResult }: PredictFormProps) {
  const [state, formAction] = useActionState(
    async (prevState: PredictState | null, formData: FormData) => {
      const result = await predictAction(prevState, formData);
      onResult?.(result);
      return result;
    },
    null,
  );

  return (
    <Card>
      <CardHeader>
        <CardTitle>BIM 객체 데이터 입력</CardTitle>
        <CardDescription>
          BIM 객체의 속성을 입력하면 KBIMS 부위코드를 예측합니다.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form action={formAction} className="space-y-4">
          {formFields.map((field) => (
            <div key={field.name} className="space-y-2">
              <Label htmlFor={field.name}>{field.label}</Label>
              <Input
                id={field.name}
                name={field.name}
                placeholder={field.placeholder}
                defaultValue={
                  state?.input?.[field.name as keyof typeof state.input] ?? ''
                }
                required
              />
            </div>
          ))}
          <SubmitButton />
        </form>
      </CardContent>
    </Card>
  );
}
